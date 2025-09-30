from typing import Optional

import json
import copy
import requests
import pydantic

from enum import Enum

from yadc.core import logging
from yadc.core import DatasetImage

from .base import BaseAPICaptioner
from .session import Session
from .utils import ErrorNormalizationMixin, ThinkingMixin

from .types import (
    GeminiModelsResponse,
    GeminiModel,
    GeminiContentResponse,
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

class SafetySettings(pydantic.BaseModel):
    data: Optional[list['SafetySetting']] = None

    @pydantic.model_validator(mode='after')
    def validate_(self):
        existing_categories = set()

        if self.data is None:
            return self

        for setting in self.data:
            if setting.category in existing_categories:
                raise ValueError(f'duplicate safetty setting: {setting.category}')
            
            existing_categories.add(setting.category)
        
        return self

class SafetySetting(pydantic.BaseModel):
    category: str
    threshold: str

    @pydantic.model_validator(mode='after')
    def validate_(self):
        try:
            assert self.category in (
                'HARM_CATEGORY_HARASSMENT',
                'HARM_CATEGORY_HATE_SPEECH',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                'HARM_CATEGORY_DANGEROUS_CONTENT',
            ), 'invalid safety setting category: refer to documentation at https://ai.google.dev/api/generate-content#v1beta.HarmCategory'

            assert self.threshold in (
                'BLOCK_LOW_AND_ABOVE',
                'BLOCK_MEDIUM_AND_ABOVE',
                'BLOCK_ONLY_HIGH',
                'BLOCK_NONE',
                'OFF',
            ), f'invalid safety setting threshold for {self.category}: refer to documentation at https://ai.google.dev/api/generate-content#HarmBlockThreshold'
        except AssertionError as e:
            raise ValueError(e)

        return self


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

class GeminiCaptioner(BaseAPICaptioner, ErrorNormalizationMixin, ThinkingMixin):
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
                - `session` (requests.Session, options): Override the session for the API calls

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

        self._session.headers = { 'x-goog-api-key': self._api_token }
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

    def load_model(self, model_repo: str, **kwargs) -> None:
        try:
            self._load_model(model_repo, **kwargs)
        except requests.HTTPError as e:
            raise ValueError(self._normalize_error(e))
        except requests.ConnectionError:
            raise ValueError(f'api unavailable: {self._api_url}')
        except AssertionError as e:
            raise ValueError(str(e))

    def _load_model(self, model_repo: str, **kwargs) -> None:
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
                assert isinstance(models_resp_json, dict), "bad model response"

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

        try:
            conversation_overrides = kwargs.pop('conversation_overrides', {})
            assert isinstance(conversation_overrides, dict), f'bad value for conversation_overrides/advanced settings; expected a dict, got: {type(conversation_overrides)}'
            
            conversation_overrides = copy.deepcopy(conversation_overrides)
            
            # just make sure this is not overridden
            conversation_overrides.pop('contents', None)

            system_role = conversation_overrides.pop('system_role', None) or 'system'
            assert isinstance(system_role, str), f'bad value for conversation_overrides/advanced settings system_role; expected a str, got: {type(system_role)}'

            user_role = conversation_overrides.pop('user_role', None) or 'user'
            assert isinstance(user_role, str), f'bad value for conversation_overrides/advanced settings user_role; expected a str, got: {type(user_role)}'

            assistant_role = conversation_overrides.pop('assistant_role', None) or 'model'
            assert isinstance(assistant_role, str), f'bad value for conversation_overrides/advanced settings assistant_role; expected a str, got: {type(assistant_role)}'

            assistant_prefill = conversation_overrides.pop('assistant_prefill', '')
            assert isinstance(assistant_prefill, str), f'bad value for conversation_overrides/advanced settings assistant_prefill; expected a str, got: {type(assistant_prefill)}'

            try:
                safety_settings_overrides = SafetySettings(data=conversation_overrides.get('safety_settings', None))
            except pydantic.ValidationError as e:
                raise AssertionError(f'bad value for conversation_overrides/advanced settings safety settings: {e}')
            
            generation_config_overrides = conversation_overrides.get('generation_config', {})
            assert isinstance(generation_config_overrides, dict), f'bad value for conversation_overrides/advanced settings generation_config; expected a dict, got: {type(generation_config_overrides)}'
        except AssertionError as e:
            raise ValueError(e)

        max_tokens = kwargs.pop('max_new_tokens', 512)

        generation_config = {
            'maxOutputTokens': max_tokens,
            'responseModalities': [ 'TEXT' ],
        }

        if self._image_quality == 'low':
            generation_config['mediaResolution'] = 'MEDIA_RESOLUTION_LOW'
        elif self._image_quality == 'medium':
            generation_config['mediaResolution'] = 'MEDIA_RESOLUTION_MEDIUM'
        elif self._image_quality == 'high':
            generation_config['mediaResolution'] = 'MEDIA_RESOLUTION_HIGH'

        conversation = {
            'system_instruction': {
                'parts': [
                    { 'text': system_prompt },
                ],
            },
            'contents': [{
                'role': user_role,
                'parts': [
                    { 'inline_data': { 'mime_type': mime_type, 'data': encoded_image } },
                    { 'text': user_prompt },
                ],
            }],
            'generationConfig': generation_config,
        }

        if assistant_prefill:
            conversation['contents'].append({
                'role': assistant_role,
                'parts': [{ 'text': assistant_prefill }],
                'is_prefill': True,
            })

        if self._is_thinking_model and self._reasoning:
            conversation['generationConfig']['thinkingConfig'] = {
                'includeThoughts': not self._reasoning_exclude_output,
                'thinkingBudget': APITypes.BASE.get_thinking_budget(self._reasoning_effort),
            }

        if safety_settings_overrides.data is not None:
            conversation['safetySettings'] = safety_settings_overrides.model_dump()['data']

        if generation_config_overrides:
            generation_config.update(generation_config_overrides)

        return conversation

    def _extract_assistant_prefill(self, conversation: dict):
        try:
            last_message = conversation['contents'][-1]
            if last_message['is_prefill']:
                assistant_prefill = last_message['parts'][0]['text']
                last_message.pop('is_prefill', None)
            else:
                assistant_prefill = ''

            assert isinstance(assistant_prefill, str)
        except:
            _logger.debug('Failed to extract assistant prefill', exc_info=True)
            assistant_prefill = ''

        return assistant_prefill

    def _generate_stream_prediction_inner(self, image: DatasetImage, **kwargs):
        assert self._current_model, "no model loaded"

        conversation = self.conversation(image, **kwargs)
        assistant_prefill = self._extract_assistant_prefill(conversation)

        with self._session.post(f'models/{self._current_model}:streamGenerateContent?alt=sse', stream=True, json=conversation) as conversation_resp:
            try:
                conversation_resp.raise_for_status()
            except:
                # NOTE: consume the stream so error can be parsed
                conversation_error = '\n'.join(conversation_resp.iter_lines(decode_unicode=True))
                conversation_error = conversation_error.strip()

                raise ErrorNormalizationMixin.GenerationError(conversation_error)

            if assistant_prefill:
                yield assistant_prefill

            conversation_error = '' # in some scenarios, gemini api will just send the error directly instead of as a sse data line
            conversation_stopped = False

            is_thinking = False   # used to wrap the thoughts in <think>...</think>
            is_prediction = False # prevents the thoughts from being printed if the first thought is done

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
                    line_response = GeminiContentResponse(**line_json)

                    if line_response.usageMetadata and line_response.responseId != "SKIPPED":
                        self._api_usage[line_response.responseId] = APIUsage(
                            response_tokens=line_response.usageMetadata.candidatesTokenCount,
                            prompt_tokens=line_response.usageMetadata.promptTokenCount,
                            total_tokens=line_response.usageMetadata.totalTokenCount,
                            thoughts_tokens=line_response.usageMetadata.thoughtsTokenCount,
                        )

                    found_candidate = False
                    for candidate in line_response.candidates:
                        if found_candidate:
                            break

                        if candidate.finishReason and candidate.finishReason != 'STOP':
                            raise ValueError(self._normalize_error(line_response))

                        for part in candidate.content.parts:
                            if text := part.text:
                                if part.thought:
                                    if is_prediction:
                                        continue

                                    if not is_thinking:
                                        yield '<think>'
                                        is_thinking = True

                                    yield text

                                    continue

                                if is_thinking:
                                    yield '</think>'
                                    is_thinking = False

                                is_prediction = True

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

                raise ErrorNormalizationMixin.GenerationError(conversation_error)

    def _generate_stream_prediction(self, image: DatasetImage, **kwargs):
        try:
            yield from self._generate_stream_prediction_inner(image, **kwargs)
            return
        except requests.HTTPError as e:
            raise ValueError(self._normalize_error(e))
        except ErrorNormalizationMixin.GenerationError as e:
            raise ValueError(self._normalize_error(e))

    def _generate_prediction(self, image: DatasetImage, **kwargs):
        assert self._current_model, "model not loaded"

        # make sure stream is not set in kwargs
        kwargs.pop('stream', None)

        conversation = self.conversation(image, **kwargs)
        assistant_prefill = self._extract_assistant_prefill(conversation)

        is_thinking = False   # used to wrap the thoughts in <think>...</think>

        with self._session.post(f'models/{self._current_model}:generateContent', stream=False, json=conversation) as conversation_resp:
            conversation_resp.raise_for_status()

            try:
                conversation_json = json.loads(conversation_resp.text)
                assert isinstance(conversation_json, dict), "api did not return valid json"
            except AssertionError as e:
                _logger.debug('Failed to decode response to json: %s', conversation_resp.text)
                raise ValueError(str(e))
            except json.JSONDecodeError as e:
                _logger.debug('Failed to decode response to json: %s', conversation_resp.text)
                raise ValueError('api did not return json')

            try:
                conversation_response = GeminiContentResponse(**conversation_json)
            except AssertionError as e:
                _logger.debug('Failed to decode response to object: %s', conversation_resp.text)
                raise ValueError(str(e))
            except pydantic.ValidationError as e:
                _logger.debug('Failed to decode response to object: %s', conversation_resp.text)
                raise ValueError('api did not return a valid response')

            if conversation_response.usageMetadata and conversation_response.responseId != "SKIPPED":
                self._api_usage[conversation_response.responseId] = APIUsage(
                    response_tokens=conversation_response.usageMetadata.candidatesTokenCount,
                    prompt_tokens=conversation_response.usageMetadata.promptTokenCount,
                    total_tokens=conversation_response.usageMetadata.totalTokenCount,
                    thoughts_tokens=conversation_response.usageMetadata.thoughtsTokenCount,
                )

            thought_buffer = ''

            for candidate in conversation_response.candidates:
                if candidate.finishReason and candidate.finishReason != 'STOP':
                    raise ValueError(self._normalize_error(conversation_response))

                for part in candidate.content.parts:
                    if text := part.text:
                        if not part.thought:
                            continue

                        if not is_thinking:
                            thought_buffer += '<think>'
                            is_thinking = True
                        
                        thought_buffer += text

                if is_thinking:
                    thought_buffer += '</think>'
                    is_thinking = False

                for part in candidate.content.parts:
                    if text := part.text:
                        if part.thought:
                            continue

                        if assistant_prefill:
                            text = assistant_prefill + text

                        return thought_buffer + text

            raise ValueError('api did not return text')


    def predict(self, image: DatasetImage, **kwargs):
        try:
            return self._handle_thinking(self._generate_prediction(image, **kwargs))
        except requests.HTTPError as e:
            raise ValueError(self._normalize_error(e))
    
    def predict_stream(self, image: DatasetImage, **kwargs):
        yield from self._handle_thinking_streaming(self._generate_stream_prediction(image, **kwargs))
