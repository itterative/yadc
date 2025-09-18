from typing import Generator

import io
import json
import base64
import requests

from enum import Enum
from PIL import Image

from yadc.core import Captioner, DatasetImage

from .session import Session
from .types import GeminiModelsResponse, GeminiModel

class APITypes(str, Enum):
    BASE = 'base'

    def __str__(self) -> str:
        return self.value

    @property
    def max_image_size(self):
        return (2048, 2048)

    @property
    def max_image_encoded_size(self):
        return 5 * 1024 * 1024

class GeminiCaptioner(Captioner):
    _current_model: str|None = None
    _is_thinking_model: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._api_url: str = kwargs.pop('api_url', '')
        self._api_token: str = kwargs.pop('api_token', '')
        self._image_quality: str = kwargs.pop('image_quality', 'auto')

        if not self._api_url:
            raise ValueError("no api_url")

        if not self._api_token:
            raise ValueError("no api_token")

        self._session = Session(self._api_url, headers={ 'x-goog-api-key': self._api_token })


    def load_model(self, model_repo: str, **kwargs) -> None:
        model_repo = model_repo.removeprefix('models/')

        with self._session.get(f'/v1beta/models/{model_repo}') as model_resp:
            if model_resp.ok:
                model_resp_json = model_resp.json()
                model = GeminiModel(**model_resp_json)

                if 'generateContent' not in model.supportedGenerationMethods:
                    raise ValueError(f"model {model_repo} does not have generative capabilities: available capabilities: {', '.join(model.supportedGenerationMethods)}")
                
                self._is_thinking_model = model.thinking
                self._current_model = model.name.removeprefix('models/')
                print(f'Model set to {self._current_model}.')

                return

        available_models: list[str] = []
        model_found = False
        next_token: str|None = None

        model_repo_prefixed = f'models/{model_repo}'

        while not model_found:
            if next_token is not None:
                models_url_path = f'/v1beta/models?pageToken={next_token}'
            else:
                models_url_path = f'/v1beta/models'

            with self._session.get(models_url_path) as models_resp:
                models_resp.raise_for_status()

                models_resp_json = models_resp.json()
                models = GeminiModelsResponse(**models_resp_json)

                for model in models.models:
                    if 'generateContent' not in model.supportedGenerationMethods:
                        continue

                    if model_repo_prefixed == model.name or model_repo == model.name:
                        self._is_thinking_model = model.thinking
                        self._current_model = model.name.removeprefix('models/')
                        break
                    
                    available_models.append(model.name.removeprefix('models/'))

                next_token = models.nextPageToken

                if next_token is None:
                    break

        if not self._current_model:
            if available_models:
                raise ValueError(f'model not found: {model_repo}; available models: {", ".join(available_models)}')

            raise ValueError(f'model not found: {model_repo}; no models available')
        
        print(f'Model set to {self._current_model}.')

    def unload_model(self) -> None:
        pass

    def offload_model(self) -> None:
        pass


    def encode_image(self, image: DatasetImage, **kwargs):
        call_depth = kwargs.pop('call_depth', 0)
        assert isinstance(call_depth, int), f"encode_image called with bad call_depth type: {type(call_depth)}"
        assert call_depth < 5, f"encode_image reached maximum call depth"

        image_format = kwargs.pop('image_format', 'PNG')
        assert isinstance(image_format, str), f"encode_image called with bad image_format type: {type(image_format)}"

        image_format = image_format.upper()
        assert image_format in ('JPEG', 'PNG'), f"encode_image called with bad image_format: only JPEG or PNG is allowed"

        image_quality = kwargs.pop('image_quality', None)
        assert image_quality is None or isinstance(image_quality, int), f"encode_image called with bad image_quality type: {type(image_quality)}"
        assert image_quality is None or (image_quality > 10 and image_quality <= 100), f"encode_image called with bad image_quality: {image_quality}"

        buffer = io.BytesIO()
        image_obj = image.read_image()

        # resize image if too large
        image_obj.thumbnail(APITypes.BASE.max_image_size, Image.Resampling.LANCZOS)

        if image_format == 'JPEG' and image_obj.mode in ('RGBA', 'LA'):
            image_composite = Image.new('RGB', image_obj.size, (255, 255, 255))
            image_composite.paste(image_obj, mask=image_obj.split()[-1])
            image_obj = image_composite

        image_obj.save(buffer, format=image_format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        if len(encoded_image) > APITypes.BASE.max_image_encoded_size:
            # start at lossless, then degrade with each iteration
            image_quality = 100 if image_quality is None else image_quality - 10

            return self.encode_image(image, image_format='JPEG', call_depth=call_depth+1, image_quality=image_quality, **kwargs)

        return f'image/{image_format.lower()}', base64.b64encode(buffer.getvalue()).decode('utf-8')

    def conversation(self, image: DatasetImage, **kwargs):
        system_prompt, user_prompt = self._prompts_from_image(image, **kwargs)

        mime_type, encoded_image = self.encode_image(image, **kwargs)

        temperature = kwargs.pop('temperature', 0.8)
        top_p = kwargs.pop('top_p', 0.9)
        top_k = kwargs.pop('top_k', 64)
        max_tokens = kwargs.pop('max_new_tokens', 512)

        generation_config = {
            'maxOutputTokens': max_tokens,
            'temperature': temperature,
            'topP': top_p,
            'topK': top_k,
            'responseModalities': [ 'TEXT' ],
        }

        if self._image_quality == 'low':
            generation_config['mediaResolution'] = 'MEDIA_RESOLUTION_LOW'
        elif self._image_quality == 'high':
            generation_config['mediaResolution'] = 'MEDIA_RESOLUTION_HIGH'

        conversation = {
            'system_instruction': {
                'parts': [
                    { 'text': system_prompt },
                ],
            },
            'contents': [{
                'parts': [
                    { 'inline_data': { 'mime_type': mime_type, 'data': encoded_image } },
                    { 'text': user_prompt },
                ],
            }],
            'generationConfig': generation_config,
        }

        if self._is_thinking_model:
            conversation['generationConfig']['thinkingConfig'] = {
                'includeThoughts': False,
                'thinkingBudget': 0,
            }

        return conversation
   

    def _generate_prediction(self, image: DatasetImage, **kwargs):
        assert self._current_model, "no model loaded"

        conversation = self.conversation(image, **kwargs)

        with self._session.post(f'/v1beta/models/{self._current_model}:streamGenerateContent?alt=sse', stream=True, json=conversation) as conversation_resp:
            try:
                conversation_resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print('body:', e.response.text)
                raise

            conversation_stopped = False

            for line in conversation_resp.iter_lines(decode_unicode=True):
                assert isinstance(line, str)

                if not line or conversation_stopped:
                    continue

                try:
                    line = line.removeprefix('data:')

                    if line == '[DONE]':
                        conversation_stopped = True
                        continue

                    line_json = json.loads(line.lstrip())
                except json.JSONDecodeError as e:
                    print(f'Warning: failed to decode line: {line}')
                    continue

                try:
                    assert isinstance(line_json, dict), "not a dict"
                    assert 'candidates' in line_json, "no candidates"

                    line_json_candidates = line_json['candidates']
                    assert isinstance(line_json_candidates, list), "bad candidates"

                    for candidate in line_json_candidates:
                        assert isinstance(candidate, dict), "bad candidate"

                        condidate_content = candidate.get('content', None)
                        if not isinstance(condidate_content, dict):
                            continue

                        condidate_content_parts = condidate_content.get('parts', None)
                        if not isinstance(condidate_content_parts, list):
                            continue

                        for part in condidate_content_parts:
                            if not isinstance(part, dict):
                                pass

                            part_text = part.get('text')
                            if not isinstance(part_text, str):
                                continue

                            yield part_text
                except AssertionError as e:
                    print(f'Error: failed to process line: {e}: {line}')
                    break

    def predict(self, image: DatasetImage, **kwargs):
        return ''.join(list(self._generate_prediction(image, **kwargs)))
    
    def predict_stream(self, image: DatasetImage, **kwargs):
        yield from self._generate_prediction(image, **kwargs)
