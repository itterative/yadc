
import time
import json
import requests
import pydantic

from enum import Enum
from urllib.parse import urlparse

from yadc.core import logging
from yadc.core import DatasetImage

from .base import BaseAPICaptioner
from .session import Session
from .mixins import ErrorNormalizationMixin

from .types import (
    OpenAIModelsResponse,
    OpenAIStreamingResponse,
    OpenRouterCreditsResponse,
    KoboldAdminSettingsReponse,
    KoboldAdminReloadModelReponse,
    KoboldAdminCurrentModelResponse,
    KoboldServiceInfoResponse,
)

_logger = logging.get_logger(__name__)

class APITypes(str, Enum):
    OPENAI = 'openai'
    OPENROUTER = 'openrouter'
    KOBOLDCPP = 'koboldcpp'
    VLLM = 'vllm'
    OLLAMA = 'ollama'

    def __str__(self) -> str:
        return self.value
    
    @property
    def max_image_size(self):
        match self:
            case APITypes.OPENAI: return (1024, 1024)
            case APITypes.OPENROUTER: return (1024, 1024)
            case _: return (1536, 1536) # slightly increased for local backends
    
    @property
    def max_image_encoded_size(self):
        match self:
            case APITypes.OPENAI: return 10 * 1024 * 1024
            case APITypes.OPENROUTER: return 10 * 1024 * 1024
            case _: return 25 * 1024 * 1024 # slightly increased for local backends

class APIUsage:
    def __init__(
        self,
        prompt_tokens: int,
        response_tokens: int,
        total_tokens: int,
        thoughts_tokens: int,
    ):
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.total_tokens = total_tokens
        self.thoughts_tokens = thoughts_tokens

class OpenAICaptioner(BaseAPICaptioner, ErrorNormalizationMixin):
    """
    Implementation for image captioning models using OpenAI-compatible endpoints.

    Required API credentials:
    - `api_token`: Your OpenAI API key (not required not unathenticated APIs)

    Example:
    ```
        captioner = OpenAICaptioner(
            api_url="https://api.openai.com",
            api_token="your-api-key"
        )
        captioner.load_model("gpt-5-mini")
        caption = captioner.predict(dataset_image)
    ```
    """

    _current_model: str|None = None

    def __init__(self, **kwargs):
        """
        Initializes the OpenAICaptioner with API and template configuration.

        Args:
            api_url (str): Base URL for the OpenAI API endpoint.

            **kwargs: Optional keyword arguments:
                - `api_token` (str): OpenAI API key for authentication.
                - `prompt_template_name` (str): Filename of the Jinja2 template to use (default: 'default.jinja').
                - `prompt_template` (str): Direct template string to override file-based templates.
                - `image_quality` (str): Quality setting for encoded images ('auto', 'low', 'high').
                - `reasoning` (bool): Enable internal chain-of-thought / extra reasoning behavior.
                - `reasoning_effort` (str, optional): Level of reasoning effort to request when `reasoning` is True ('low', 'medium', 'high'). 
                - `reasoning_exclude_output` (bool, optional): When True, exclude internal reasoning output from the caption.
                - `session` (requests.Session, options): Override the session for the API calls

        Raises:
            ValueError: If `api_url` is not provided.
        """

        super().__init__(**kwargs)

        self._api_url: str = kwargs.pop('api_url', '')
        self._api_token: str = kwargs.pop('api_token', '')
        self._store_conversation: bool = kwargs.pop('store_conversation', False)
        self._image_quality: str = kwargs.pop('image_quality', 'auto')

        self._reasoning: bool = kwargs.pop('reasoning', False)
        self._reasoning_effort: str = kwargs.pop('reasoning_effort', 'low')
        self._reasoning_exclude_output: bool = kwargs.pop('reasoning_exclude_output', True)

        if not self._api_url:
            raise ValueError("no api_url")

        session_headers = {}

        if self._api_token:
            session_headers['Authorization'] = f'Bearer {self._api_token}'
        else:
            _logger.warning('Warning: no api_token is set, requests will fail if api uses authentication')

        session: requests.Session|None = kwargs.pop('session', None)
        assert session is None or isinstance(session, requests.Session)

        self._session = Session(self._api_url, headers=session_headers, session=session)

        self._api_type = self._infer_api_type()
        _logger.info('API set to %s.', self._api_type)

        self._log_api_information()
        self._api_usage: dict[str, APIUsage] = {}

    def log_usage(self):
        usage = APIUsage(prompt_tokens=0, response_tokens=0, total_tokens=0, thoughts_tokens=0)

        for response_usage in self._api_usage.values():
            usage.prompt_tokens += response_usage.prompt_tokens
            usage.response_tokens += response_usage.response_tokens
            usage.total_tokens += response_usage.total_tokens
            usage.thoughts_tokens += response_usage.thoughts_tokens

        if usage.total_tokens == 0:
            return
        
        if usage.thoughts_tokens == 0:
            _logger.info('Used a total of %d tokens (prompt: %d, response: %d).', usage.total_tokens, usage.prompt_tokens, usage.response_tokens)
        else:
            _logger.info('Used a total of %d tokens (prompt: %d, response: %d, reasoning: %d).', usage.total_tokens, usage.prompt_tokens, usage.response_tokens, usage.thoughts_tokens)

    def _infer_api_type(self):
        # early exit

        try:
            api_url = urlparse(self._api_url)

            if api_url.netloc == 'api.openai.com':
                return APITypes.OPENAI
            
            if api_url.netloc == 'openrouter.ai':
                return APITypes.OPENROUTER
        except:
            pass

        # NOTE: might be worth to infer based on the model list
        # 
        # kobold: models are prefixed with 'koboldcpp/'
        #         and owned_by set to 'koboldcpp'
        # vllm: owned_by set to 'vllm'
        # ollama: owned_by set to ollama user or 'library'

        try:
            # koboldcpp has well defined endpoint
            #
            # reference: https://lite.koboldai.net/koboldcpp_api#/serviceinfo/get__well_known_serviceinfo

            with self._session.get('/.well-known/serviceinfo') as koboldcpp_resp:
                assert koboldcpp_resp.ok

                koboldcpp_json = koboldcpp_resp.json()
                assert isinstance(koboldcpp_json, dict)

                koboldcpp_service_info = KoboldServiceInfoResponse(**koboldcpp_json)

                assert koboldcpp_service_info.software.name.lower() == 'koboldcpp'

                return APITypes.KOBOLDCPP
        except AssertionError:
            pass
        except requests.JSONDecodeError:
            pass
        except pydantic.ValidationError:
            pass

        # the following aren't very good checks (might not work in the future)

        try:
            # vllm doesn't list their apis clearly
            # so, check for some found in their code
            #
            # /ping: https://github.com/vllm-project/vllm/blob/9fac6aa30b669de75d8718164cd99676d3530e7d/vllm/entrypoints/openai/api_server.py#L365
            # /version: https://github.com/vllm-project/vllm/blob/9fac6aa30b669de75d8718164cd99676d3530e7d/vllm/entrypoints/openai/api_server.py#L465

            with self._session.get('/ping') as vllm_resp:
                assert vllm_resp.ok

            with self._session.post('/ping') as vllm_resp:
                assert vllm_resp.ok

            with self._session.get('/version') as vllm_resp:
                assert vllm_resp.ok

                vllm_json = vllm_resp.json()
                assert isinstance(vllm_json, dict)
                assert 'version' in vllm_json

            return APITypes.VLLM
        except AssertionError:
            pass
        except requests.JSONDecodeError:
            pass

        try:
            # ollama doesn't have a well defined endpoint
            # so just use one that their docs
            #
            # reference: https://github.com/ollama/ollama/blob/main/docs/api.md#version

            with self._session.get('/api/version') as ollama_resp:
                assert ollama_resp.ok

                ollama_json = ollama_resp.json()
                assert isinstance(ollama_json, dict)
                assert 'version' in ollama_json

            return APITypes.OLLAMA
        except AssertionError:
            pass
        except requests.JSONDecodeError:
            pass

        return APITypes.OPENAI

    def _log_api_information(self):
        if self._api_type == APITypes.OPENROUTER:
            with self._session.get('credits') as credits_resp:
                try:
                    credits_resp.raise_for_status()

                    credits_resp_json = credits_resp.json()
                    assert isinstance(credits_resp_json, dict)

                    credits = OpenRouterCreditsResponse(**credits_resp_json).data
                    _logger.info('You have used %.2f out of %.2f credits with this api token.', credits.total_usage, credits.total_credits)
                except:
                    _logger.warning('Warning: failed to retrieve current credits for api token.')


    def load_model(self, model_repo: str, **kwargs) -> None:
        try:
            if self._api_type == APITypes.KOBOLDCPP:
                self._load_model_koboldcpp(model_repo)
            else:
                self._load_model_openai(model_repo)
        except requests.HTTPError as e:
            raise ValueError(self._normalize_error(e))
        except requests.ConnectionError:
            raise ValueError(f'api unavailable: {self._api_url}')

        _logger.info('Model set to %s.', self._current_model)

    def _load_model_openai(self, model_repo: str):
        if self._current_model == model_repo:
            return

        with self._session.get('models') as model_resp:
            model_resp.raise_for_status()

            model_resp_json = model_resp.json()
            assert isinstance(model_resp_json, dict)

            try:
                models = OpenAIModelsResponse(**model_resp_json)
                available_models: list[str] = []
            except pydantic.ValidationError as e:
                raise ValueError(f'failed to parse model list response') from e

            for model in models.data:
                available_models.append(model.id)

                if model.id == model_repo:
                    self._current_model = model_repo

            if not self._current_model:
                if len(available_models) > 5:
                    raise ValueError(f'model not found: {model_repo}; available models: {", ".join(available_models[:5])}, +{len(available_models)-1} more')

                if available_models:
                    raise ValueError(f'model not found: {model_repo}; available models: {", ".join(available_models)}')

                raise ValueError(f'model not found: {model_repo}; no models available')
            

    def _load_model_koboldcpp(self, model_repo: str, timeout: float = 60):
        if self._current_model == model_repo:
            return
        
        # koboldcpp api doesn't expose enough information to make sure that the right settings are already loaded
        # so we have to add some extra prefixes (e.g. koboldcpp/MODEL) and suffixes (e.g. MODEL.kcpps; assumes there is a matching kcpps file)

        # the logic goes as following
        #   1. if the openai endpoint shows the model is already loaded, use that
        #   2. if the kobold endpoint shows the model is already loaded, use that
        #   3. otherwise, unload the model, then wait for it to be loaded

        try:
            # early exit if the model is already loaded
            self._load_model_openai(model_repo)
            return
        except ValueError:
            pass

        model_koboldcpp = 'koboldcpp/' + model_repo
        model_kcpss = model_repo + '.kcpps'

        # early exit if already loaded
        with self._session.get('/api/v1/model') as model_current_resp:
            assert model_current_resp.ok

            model_current_resp_json = model_current_resp.json()
            assert isinstance(model_current_resp_json, dict)

            model_current = KoboldAdminCurrentModelResponse(**model_current_resp_json)

            if model_current.result == model_repo or model_current.result == model_koboldcpp:
                self._current_model = model_current.result
                return

        with self._session.get('/api/admin/list_options') as model_options_resp:
            assert model_options_resp.ok

            model_options_resp_json = model_options_resp.json()
            assert isinstance(model_options_resp_json, list)

            models = KoboldAdminSettingsReponse(data=model_options_resp_json)
            available_models: list[str] = []

            for model in models.data:
                if model == "unload_model":
                    continue

                available_models.append(model)

                if model == model_repo or model == model_kcpss:
                    self._current_model = model

            if not self._current_model:
                if available_models:
                    raise ValueError(f'model not found: {model_repo}; available models: {", ".join(available_models)}')

                raise ValueError(f'model not found: {model_repo}; no models available')

        with self._session.post('/api/admin/reload_config', json={"filename": self._current_model}) as model_reload_resp:
            assert model_reload_resp.ok

            model_reload_resp_json = model_reload_resp.json()
            assert isinstance(model_reload_resp_json, dict)

            if not KoboldAdminReloadModelReponse(**model_reload_resp_json).success:
                raise ValueError(f'failed to load model: {model_repo}')

        start_t = time.time()
        end_t = start_t + timeout

        # time.sleep(5) # NOTE: the api should be down within 5s; after that we keep checking which model is loaded

        while time.time() < end_t:
            try:
                with self._session.get('/api/v1/model') as model_current_resp:
                    assert model_current_resp.ok

                    model_current_resp_json = model_current_resp.json()
                    assert isinstance(model_current_resp_json, dict)

                    model_current = KoboldAdminCurrentModelResponse(**model_current_resp_json)

                    if model_current.result == "inactive":
                        time.sleep(0.5)
                        continue

                    self._current_model = model_current.result
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
                continue
        else:
            raise TimeoutError(f'failed to load model in time: {model_repo}')


    def unload_model(self):
        try:
            if self._api_type == APITypes.KOBOLDCPP:
                with self._session.post('/api/admin/reload_config', json={"filename": "unload_model"}) as unload_model_resp:
                    assert unload_model_resp.ok
        except AssertionError as e:
            raise ValueError("failed to unload model") from e

    def offload_model(self):
        pass

    def conversation(self, image: DatasetImage, **kwargs):
        system_prompt, user_prompt = self._prompts_from_image(image, **kwargs)

        mime_type, encoded_image = self._encode_image(
            image,
            max_image_size=self._api_type.max_image_size,
            max_image_encoded_size=self._api_type.max_image_encoded_size,
            **kwargs
        )

        temperature = kwargs.pop('temperature', 0.7)
        top_p = kwargs.pop('top_p', 0.95)
        top_k = kwargs.pop('top_k', 64)
        max_tokens = kwargs.pop('max_new_tokens', 512)

        conversation = {
            'model': self._current_model,
            'metadata': { 'topic': 'yadt:caption' },
            'temperature': temperature,
            'stream': True,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k,
            'store': self._store_conversation,
            'messages': [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}",
                            },
                            "detail": self._image_quality,
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ],
        }

        if self._reasoning or True:
            conversation['reasoning'] = {
                'effort': self._reasoning_effort,
                'exclude': self._reasoning_exclude_output,
            }

        # reference: https://github.com/LostRuins/koboldcpp/blob/575eb4095095939b016dc2e1957643ffb2dbf086/tools/server/bench/script.js#L98
        if self._api_type == APITypes.KOBOLDCPP:
            conversation['stop'] = ['<|im_end|>']
        elif self._api_type == APITypes.OPENROUTER:
            conversation['user'] = self._session.user_agent

        return conversation

    def _generate_prediction_inner(self, image: DatasetImage, **kwargs):
        assert self._current_model, "model not loaded"

        conversation = self.conversation(image, **kwargs)

        with self._session.post('chat/completions', stream=True, json=conversation) as conversation_resp:
            try:
                conversation_resp.raise_for_status()
            except:
                # NOTE: consume the stream so error can be parsed
                conversation_error = '\n'.join(conversation_resp.iter_lines(decode_unicode=True))
                conversation_error = conversation_error.strip()

                raise ErrorNormalizationMixin.GenerationError(conversation_error)

            converation_stopped = False

            for line in conversation_resp.iter_lines():
                # NOTE: decode_unicode option doesn't seem to work properly for some characters
                assert isinstance(line, bytes)
                line = line.decode()

                if not line or converation_stopped:
                    continue

                try:
                    # skip keepalive comments (https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation)
                    if line.startswith(':'):
                        continue

                    line = line.removeprefix('data:').strip()

                    if line == '[DONE]':
                        converation_stopped = True
                        continue

                    line_json = json.loads(line)
                except json.JSONDecodeError as e:
                    _logger.warning('Warning: failed to decode line: %s', line)
                    continue

                try:
                    assert isinstance(line_json, dict), "not a dict"
                    line_response = OpenAIStreamingResponse(**line_json)

                    if line_response.usage and line_response.id != "SKIPPED":
                        self._api_usage[line_response.id] = APIUsage(
                            response_tokens=line_response.usage.completion_tokens,
                            prompt_tokens=line_response.usage.prompt_tokens,
                            total_tokens=line_response.usage.total_tokens,
                            thoughts_tokens=0 if not line_response.usage.completion_tokens_details else line_response.usage.completion_tokens_details.reasoning_tokens,
                        )

                    if line_response.error:
                        raise ValueError(self._normalize_error(line_response))

                    for choice in line_response.choices:
                        if choice.finish_reason and choice.finish_reason != 'stop':
                            raise ValueError(self._normalize_error(line_response))

                        if content := choice.delta.content:
                            content = content.replace('◁', '<')
                            content = content.replace('▷', '>\n')

                            yield content
                            break # only retrieve first choice
                except pydantic.ValidationError:
                    _logger.error('Error: failed to process line: not a stream response: %s', line)
                    break
                except AssertionError as e:
                    _logger.error('Error: failed to process line: %s: %s', e, line)
                    break

    def _generate_prediction(self, image: DatasetImage, **kwargs):
        try:
            yield from self._generate_prediction_inner(image, **kwargs)
            return
        except requests.HTTPError as e:
            raise ValueError(self._normalize_error(e))
        except ErrorNormalizationMixin.GenerationError as e:
            raise ValueError(self._normalize_error(e))


    def predict(self, image: DatasetImage, **kwargs):
        return ''.join(list(self._generate_prediction(image, **kwargs)))
    
    def predict_stream(self, image: DatasetImage, **kwargs):
        yield from self._generate_prediction(image, **kwargs)
