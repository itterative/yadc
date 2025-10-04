from typing import Optional, TYPE_CHECKING

import os
import gc
import copy
import queue
from concurrent.futures import ThreadPoolExecutor

import toml

import torch

from transformers import AutoProcessor, Gemma3nForConditionalGeneration, Gemma3nProcessor
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

from yadc.core import logging
from yadc.core import Captioner
from yadc.core.dataset import DatasetImage

from yadc.cmd import app as yadc_app

GEMMA_REPOS = [
    "google/gemma-3n-E2B-it",
    "google/gemma-3n-E4B-it",
]

GEMMA_MODULE_COMPILATION = {
    'TimmWrapperModel': {
        'mode': 'reduce-overhead',
        'fullgraph': True,
    },
    'Gemma3nTextModel': {
        'mode': 'default',
        'fullgraph': True,
    },
    'Gemma3nMultimodalEmbedder': {
        'mode': 'default',
        'fullgraph': True,
    },
    'Gemma3nAudioEncoder': {
        'mode': 'reduce-overhead',
        'fullgraph': True,
        'dynamic': True,
    }
}

QUANTIZATION_METHODS = [
    'none',
    'quanto:int8',
]

_logger = logging.get_logger(__name__)

class _GenerationStoppingCriteria(StoppingCriteria):
    stopped: bool = False

    def __call__(self, *args, **kwargs): # type: ignore
        return self.stopped
    
class _MaxNewTokensStoppingCriterian(StoppingCriteria):
    def __init__(self, max_new_tokens: int):
        self.tokens_left = max_new_tokens

    def __call__(self, *args, **kwargs): # type: ignore
        self.tokens_left -= 1
        return self.tokens_left <= 0

def _sha256(string: str):
    import hashlib
    return hashlib.sha256(string.encode()).hexdigest()

class Gemma3nCaptioner(Captioner):
    _processor: Optional['Gemma3nProcessor'] = None
    _model: Optional['Gemma3nForConditionalGeneration'] = None
    _cache: StaticCache|None = None
    _last_image: DatasetImage|None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._torch_compile = kwargs.pop('_torch_compile', True)

        try:
            self._max_tokens = int(kwargs.pop('max_tokens', 8096))
        except:
            raise ValueError(f'bad max_tokens value')

        self._quantization: str = kwargs.pop('quantization', 'quanto:int8')

        try:
            assert isinstance(self._quantization, str)
            assert self._quantization in QUANTIZATION_METHODS
        except AssertionError:
            raise ValueError(f'bad quantization: {self._quantization}; available quant methods: {", ".join(QUANTIZATION_METHODS)}')

        try:
            pass
        except:
            pass

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Thread-gemma3n-')

        self._model_repo: Optional[str] = None

    def load_model(self, model_repo: str, **kwargs) -> None:
        if self._model_repo == model_repo:
            return

        if not model_repo:
            raise ValueError(f'repo required for Gemma 3n')
        elif model_repo not in GEMMA_REPOS:
            raise ValueError(f'bad repo for Gemma 3n: {model_repo}')
        
        self._model_repo = model_repo
        
        match self._quantization:
            case 'quanto:int8':
                from transformers import QuantoConfig

                quantization_config = QuantoConfig(
                    weights="int8",
                    modules_to_not_convert=["correction_coefs", "prediction_coefs", "lm_head", "embedding_projection"],
                )

            case 'none':
                quantization_config = None

            case _:
                raise ValueError(f'quantization not supported: {self._quantization}')

        self._processor = AutoProcessor.from_pretrained(model_repo, use_fast=True)
        self._model = Gemma3nForConditionalGeneration.from_pretrained(
            model_repo,
            device_map="auto",
            dtype="auto",
            quantization_config=quantization_config,
        ).eval()

        self._model.requires_grad_(False)
        self._load_torch_compile_cache()

    def unload_model(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._model_repo = None
        self._processor = None
        self._cache = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_model(self) -> None:
        if self._model is None:
            return
        
        self._model = self._model.to("cpu") # type: ignore

    def _torch_compile_info(self):
        assert self._model_repo

        model_repo = self._model_repo.replace('/', '--')
        model_cache_dir = yadc_app.CACHE_PATH / 'torch_compile' / model_repo
        model_cache_info = self._compile_nn_module_cache_info()
        model_cache_info_dir = _sha256(model_cache_info)

        model_cache_data = model_cache_dir / model_cache_info_dir / 'data'
        model_cache_meta = model_cache_dir / model_cache_info_dir / 'meta'

        return model_cache_meta, model_cache_data, model_cache_info

    def _load_torch_compile_cache(self):
        if not self._torch_compile:
            _logger.debug('Not using torch.compile')
            return
        
        assert self._model

        _logger.debug('Started model optimization.')

        model_cache_meta, model_cache_data, compile_kwargs = self._torch_compile_info()

        if model_cache_meta.exists() and model_cache_data.exists():
            if model_cache_meta.read_text() != compile_kwargs:
                _logger.debug('Invalidated torch.compile cache: %s', self._model_repo)
            else:
                _logger.debug('Loading torch compilation cache...')
                torch.compiler.load_cache_artifacts(model_cache_data.read_bytes())
                _logger.debug('Loaded torch compilation cache.')
        else:
            _logger.warning('Warning: torch compilation is enabled. No cached compilation results found. This will take a long time...')

        torch.set_float32_matmul_precision('high')

        _logger.debug('Compiling modules...')
        self._compile_nn_module(self._model)
        _logger.debug('Compiled modules.')

        _logger.debug('Finished model optimization.')
    
    def _save_torch_compile_cache(self):
        if not self._torch_compile:
            return
        
        if getattr(self, '__torch_compile_artifacts_saved', False):
            return

        model_cache_meta, model_cache_data, compile_kwargs = self._torch_compile_info()

        # if model_cache_meta.exists() and model_cache_meta.read_text() == compile_kwargs:
        #     return

        torch_compile_artifacts = torch.compiler.save_cache_artifacts()

        if torch_compile_artifacts is None:
            return
        
        if model_cache_data.exists() and model_cache_data.read_bytes() == torch_compile_artifacts[0]:
            return

        model_cache_data.parent.mkdir(mode=0o750, exist_ok=True)

        model_cache_meta.write_text(self._compile_nn_module_cache_info())
        model_cache_data.write_bytes(torch_compile_artifacts[0])

        setattr(self, '__torch_compile_artifacts_saved', True)

    def _compile_nn_module_cache_info(self):
        return toml.dumps({
            'version': '1',
            'model_repo': self._model_repo,
            'max_tokens': self._max_tokens,
            'quantization': self._quantization,
            'compiled_modules': GEMMA_MODULE_COMPILATION,
        })

    def _compile_nn_module(self, module: torch.nn.Module):
        for name, c_module in module.named_children():
            if c_module.__class__.__name__ in GEMMA_MODULE_COMPILATION:
                _logger.trace('Compiling module: %s', c_module.__class__.__name__)
                compiled_c_module = torch.compile(c_module, **GEMMA_MODULE_COMPILATION[c_module.__class__.__name__])
                setattr(module, name, compiled_c_module)
            else:
                _logger.trace('Not compiling module: %s', c_module.__class__.__name__)

            self._compile_nn_module(c_module)

    def _kv_cache(self, image: DatasetImage):
        assert self._model

        if self._cache is None:
            self._cache = StaticCache(
                config=self._model.config,
                max_cache_len=self._max_tokens,
                device=self._model.device,
                dtype=self._model.dtype,
            )

        if self._last_image is image:
            assert self._cache
            return self._cache

        if self._cache is not None:
            self._cache = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._last_image = image
        
        self._cache = StaticCache(
            config=self._model.config,
            max_cache_len=self._max_tokens,
            device=self._model.device,
            dtype=self._model.dtype,
        )

        return self._cache


    def conversation(self, image: DatasetImage, **kwargs):
        system_prompt, user_prompt = self._prompts_from_image(image, **kwargs)

        try:
            conversation_overrides = kwargs.pop('conversation_overrides', {})
            assert isinstance(conversation_overrides, dict), f'bad value for conversation_overrides/advanced settings; expected a dict, got: {type(conversation_overrides)}'

            conversation_overrides = copy.deepcopy(conversation_overrides)

            # just make sure this is not overridden
            conversation_overrides.pop('stream', None)
            conversation_overrides.pop('store', None)
            conversation_overrides.pop('messages', None)

            system_role = conversation_overrides.pop('system_role', None) or 'system'
            assert isinstance(system_role, str), f'bad value for conversation_overrides/advanced settings system_role; expected a str, got: {type(system_role)}'

            user_role = conversation_overrides.pop('user_role', None) or 'user'
            assert isinstance(user_role, str), f'bad value for conversation_overrides/advanced settings user_role; expected a str, got: {type(user_role)}'

            assistant_role = conversation_overrides.pop('assistant_role', None) or 'assistant'
            assert isinstance(assistant_role, str), f'bad value for conversation_overrides/advanced settings assistant_role; expected a str, got: {type(assistant_role)}'

            assistant_prefill = conversation_overrides.pop('assistant_prefill', '')
            assert isinstance(assistant_prefill, str), f'bad value for conversation_overrides/advanced settings assistant_prefill; expected a str, got: {type(assistant_prefill)}'
        except AssertionError as e:
            raise ValueError(e)

        max_tokens = kwargs.pop('max_new_tokens', 512)
        assert isinstance(max_tokens, int), f'bad value for max_tokens; expected int, got: {type(max_tokens)}'

        conversation = {
            'max_completion_tokens': max_tokens,
            'messages': [
                {
                    "role": system_role,
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        },
                    ]
                },
                {
                    "role": user_role,
                    "content": [
                        {
                            "type": "image",
                            "image": image.read_image(),
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ],
        }

        if assistant_prefill:
            conversation['messages'].append({
                "role": assistant_role,
                "content": [{
                    "type": "text",
                    "text": assistant_prefill
                }],
                "is_prefill": True,
            })

        conversation.update(conversation_overrides)

        for key, value in list(conversation.items()):
            if value is None:
                conversation.pop(key)

        return conversation

    def _extract_assistant_prefill(self, conversation: dict):
        assistant_prefill = ''
        try:
            last_message = conversation['messages'][-1]
            if last_message.pop('is_prefill', False):
                assistant_prefill = last_message['content'][0]['text']
            else:
                assistant_prefill = ''

            assert isinstance(assistant_prefill, str)
        except Exception:
            _logger.debug('Failed to extract assistant prefill', exc_info=True)
            assistant_prefill = ''

        return assistant_prefill

    def predict_stream(self, image: DatasetImage, **kwargs):
        try:
            assert self._processor is not None, "No processor loaded"
            assert self._model is not None, "No model loaded"
            assert self._model_repo is not None, "No model loaded"
        except AssertionError as e:
            raise ValueError(str(e))

        conversation = self.conversation(image, stream=False, **kwargs)
        assistant_prefill = self._extract_assistant_prefill(conversation)

        max_new_tokens: int = kwargs.pop("max_completion_tokens", None) or 300
        temperature: float = kwargs.pop("temperature", None) or 0.6
        top_k: float = kwargs.pop("top_p", None) or 64
        top_p: float = kwargs.pop("top_p", None) or 0.9

        inputs = self._processor.apply_chat_template(conversation["messages"], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",) # type: ignore

        # inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        inputs = inputs.to(self._model.device)

        def _inference(streamer: TextIteratorStreamer, cancel: StoppingCriteria):
            assert self._model

            with torch.inference_mode():
                generation = self._model.generate(
                    **inputs,
                    max_length=self._max_tokens,
                    do_sample=True,
                    streamer=streamer,
                    use_cache=True,
                    # past_key_values=self._kv_cache(image),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    stopping_criteria=StoppingCriteriaList([
                        cancel, _MaxNewTokensStoppingCriterian(max_new_tokens)
                    ]),
                )[0]

                generation = generation[inputs['input_ids'].shape[1]:]

        generation_criteria = _GenerationStoppingCriteria()
        streamer = TextIteratorStreamer(self._processor.tokenizer, skip_prompt=True, timeout=0.1, skip_special_tokens=True) # type: ignore

        try:
            t = self._pool.submit(_inference, streamer, generation_criteria)

            if assistant_prefill:
                yield assistant_prefill

            while True:
                try:
                    yield next(streamer)
                except queue.Empty:
                    if t.done():
                        break
                except StopIteration:
                    break

            t.result()
            self._save_torch_compile_cache()
        except KeyboardInterrupt:
            generation_criteria.stopped = True
            raise


    def predict(self, image: DatasetImage, **kwargs):
        return ''.join(self.predict_stream(image, **kwargs))
