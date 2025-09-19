import json
import requests
import pydantic

import io
import base64

from enum import Enum
from PIL import Image

from yadc.core import logging
from yadc.core import Captioner, DatasetImage

from .session import Session

from .types import (
    GeminiModelsResponse,
    GeminiModel,
    GeminiErrorResponse,
    GeminiStreamingResponse,
)

_logger = logging.get_logger(__name__)

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
    
    def get_thinking_budget(self, effort: str):
        match effort:
            case 'low': return 512
            case 'medium': return 1024
            case 'high': return 2048

            # shouldn't happen, but we should have a default
            case _: return 512

class GeminiCaptioner(Captioner):
    """
    Implementation for image captioning models using the Google Gemini API.

    Required API credentials:
    - `api_token`: Your Google AI API key (from Google Cloud Console or AI Studio)

    Example:
    ```
        captioner = GeminiCaptioner(
            api_url="https://generativelanguage.googleapis.com/v1beta",
            api_token="your-api-key"
        )
        captioner.load_model("gemini-pro-vision")
        caption = captioner.predict(dataset_image)
    ```
    """

    _current_model: str|None = None
    _is_thinking_model: bool = False

    def __init__(self, **kwargs):
        """
        Initializes the GeminiCaptioner with API and template configuration.

        Args:
            api_token (str): Google API key for authentication.

            **kwargs: Optional keyword arguments:
                - `api_url` (str): Base URL for the Gemini API endpoint.
                - `prompt_template_name` (str): Filename of the Jinja2 template to use (default: 'default.jinja').
                - `prompt_template` (str): Direct template string to override file-based templates.
                - `image_quality` (str): Quality setting for encoded images ('auto', 'low', 'high').
                - `reasoning` (bool): Enable internal chain-of-thought / extra reasoning behavior.
                - `reasoning_effort` (str, optional): Level of reasoning effort to request when `reasoning` is True ('low', 'medium', 'high'). 
                - `reasoning_exclude_output` (bool, optional): When True, exclude internal reasoning output from the caption.

        Raises:
            ValueError: If `api_token` is not provided.
        """

        super().__init__(**kwargs)

        self._api_url: str = kwargs.pop('api_url', 'https://generativelanguage.googleapis.com/v1beta')
        self._api_token: str = kwargs.pop('api_token', '')
        self._image_quality: str = kwargs.pop('image_quality', 'auto')

        self._reasoning: bool = kwargs.pop('reasoning', False)
        self._reasoning_effort: str = kwargs.pop('reasoning_effort', 'low')
        self._reasoning_exclude_output: bool = kwargs.pop('reasoning_exclude_output', True)

        if not self._api_url:
            raise ValueError("no api_url")

        if not self._api_token:
            raise ValueError("no api_token")

        self._session = Session(self._api_url, headers={ 'x-goog-api-key': self._api_token })


    def load_model(self, model_repo: str, **kwargs) -> None:
        model_repo = model_repo.removeprefix('models/')

        with self._session.get(f'models/{model_repo}') as model_resp:
            if model_resp.ok:
                model_resp_json = model_resp.json()
                model = GeminiModel(**model_resp_json)

                if 'generateContent' not in model.supportedGenerationMethods:
                    raise ValueError(f"model {model_repo} does not have generative capabilities: available capabilities: {', '.join(model.supportedGenerationMethods)}")
                
                self._is_thinking_model = model.thinking
                self._current_model = model.name.removeprefix('models/')
                _logger.info('Model set to %s.', self._current_model)

                return

        available_models: list[str] = []
        model_found = False
        next_token: str|None = None

        model_repo_prefixed = f'models/{model_repo}'

        while not model_found:
            if next_token is not None:
                models_url_path = f'models?pageToken={next_token}'
            else:
                models_url_path = f'models'

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

        _logger.info('Model set to %s.', self._current_model)

        if self._is_thinking_model and self._reasoning:
            _logger.warning('Warning: selected a model without reasoning capabilities, but reasoning is enabled.')

    def unload_model(self) -> None:
        pass

    def offload_model(self) -> None:
        pass


    def conversation(self, image: DatasetImage, **kwargs):
        system_prompt, user_prompt = self._prompts_from_image(image, **kwargs)

        mime_type, encoded_image = self._encode_image(
            image,
            max_image_size=APITypes.BASE.max_image_size,
            max_image_encoded_size=APITypes.BASE.max_image_encoded_size,
            **kwargs
        )

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

        if self._is_thinking_model and self._reasoning:
            conversation['generationConfig']['thinkingConfig'] = {
                'includeThoughts': not self._reasoning_exclude_output,
                'thinkingBudget': APITypes.BASE.get_thinking_budget(self._reasoning_effort),
            }

        return conversation
   

    def _generate_prediction_inner(self, image: DatasetImage, **kwargs):
        assert self._current_model, "no model loaded"

        conversation = self.conversation(image, **kwargs)

        with self._session.post(f'models/{self._current_model}:streamGenerateContent?alt=sse', stream=True, json=conversation) as conversation_resp:
            try:
                conversation_resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                _logger.debug('HTTP request failed. Body: %s', e.response.text)
                raise

            conversation_error = ''
            conversation_stopped = False

            for line in conversation_resp.iter_lines():
                # NOTE: decode_unicode option doesn't seem to work properly for some characters
                assert isinstance(line, bytes)
                line = line.decode()

                if not line or conversation_stopped:
                    continue

                try:
                    # skip keepalive comments (https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation)
                    if line.startswith(':'):
                        continue

                    line = line.removeprefix('data:').strip()

                    if line == '[DONE]':
                        conversation_stopped = True
                        continue

                    line_json = json.loads(line)
                except json.JSONDecodeError as e:
                    if line.startswith('{'):
                        # likely json, try parsing outside
                        conversation_error = line + '\n'
                        break

                    _logger.warning('Warning: failed to decode line: %s', line)
                    continue

                try:
                    assert isinstance(line_json, dict), "not a dict"
                    line_response = GeminiStreamingResponse(**line_json)

                    found_candidate = False
                    for candidate in line_response.candidates:
                        if found_candidate:
                            break

                        for part in candidate.content.parts:
                            if text := part.text:
                                # skip for now
                                if part.thought:
                                    continue

                                yield text

                                found_candidate = True
                                break
                except pydantic.ValidationError:
                    _logger.error('Error: failed to process line: not a stream response: %s', line)
                    break
                except AssertionError as e:
                    _logger.error('Error: failed to process line: %s: %s', e, line)
                    break

            if conversation_error:
                conversation_error += '\n'.join(conversation_resp.iter_lines(decode_unicode=True))
                conversation_error = conversation_error.strip()

                # override the response
                raise ValueError(-1, conversation_error)

    def _generate_prediction(self, image: DatasetImage, **kwargs):
        try:
            yield from self._generate_prediction_inner(image, **kwargs)
            return
        except ValueError as e:
            # TODO: better exception here
            error_code = f'sse {e.args[0]}'
            response_text = e.args[1]
        except requests.HTTPError as e:
            error_code = f'http {e.response.status_code}'
            response_text = e.response.text

        error_status = ''
        error_message = 'unknown error'

        try:
            conversation_error_json = json.loads(response_text)
            assert isinstance(conversation_error_json, dict)

            error_response = GeminiErrorResponse(**conversation_error_json).error

            error_code = str(error_response.code)
            error_status = error_response.status
            error_message = error_response.message
        except:
            _logger.warning('Warning: failed to process http error: %s', response_text)

        # some defaults if response cannot be processed
        if not error_message:
            match error_code:
                case 400: error_message = 'request could not be completed'
                case 401: error_message = 'authentication failure'
                case 402: error_message = 'not enough api credits; payment needed'
                case 429: error_message = 'overloaded'
                case 502: error_message = 'unavailable'
                case 503: error_message = 'unavailable'
                case _: error_message = 'unknown error'

        if not error_status:
            raise ValueError(f'api returned an error ({error_code}): {error_message}')

        raise ValueError(f'api returned an error ({error_status} {error_code}): {error_message}')


    def predict(self, image: DatasetImage, **kwargs):
        return ''.join(list(self._generate_prediction(image, **kwargs)))
    
    def predict_stream(self, image: DatasetImage, **kwargs):
        yield from self._generate_prediction(image, **kwargs)
