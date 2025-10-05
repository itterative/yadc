
import copy
import json
import requests
import pydantic

from enum import Enum

from yadc.core import logging
from yadc.core import DatasetImage

from .base import BaseAPICaptioner
from .utils import ErrorNormalizationMixin, ThinkingMixin

from .types import (
    OpenAIModelsResponse,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChunkResponse,
)

_logger = logging.get_logger(__name__)

CHAT_COMPLETION_OBJECT = 'chat.completion'
CHAT_COMPLETION_CHUNK_OBJECT = 'chat.completion.chunk'

class APITypes(str, Enum):
    OPENAI = 'openai'
    OPENROUTER = 'openrouter'
    LLAMACPP = 'llamacpp'
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

class OpenAICaptioner(BaseAPICaptioner, ErrorNormalizationMixin, ThinkingMixin):
    """
    Implementation for image captioning models using OpenAI-compatible endpoints.

    Required API credentials:
    - `api_token`: Your OpenAI API key (not required not unathenticated APIs)

    Example:
    ```
        captioner = OpenAICaptioner(
            api_url="https://api.openai.com/v1",
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
            api_token (str): OpenAI API key for authentication.

            **kwargs: Optional keyword arguments:
                - `prompt_template` (str): The prompt template used for captioning. If none is provided, the default will be used.
                - `image_quality` (str): Quality setting for encoded images ('auto', 'low', 'high').
                - `reasoning` (bool): Enable internal chain-of-thought / extra reasoning behavior.
                - `reasoning_effort` (str, optional): Level of reasoning effort to request when `reasoning` is True ('low', 'medium', 'high'). 
                - `reasoning_exclude_output` (bool, optional): When True, exclude internal reasoning output from the caption.
                - `session` (requests.Session, options): Override the session for the API calls

        Raises:
            ValueError: If `api_url` is not provided.
        """

        BaseAPICaptioner.__init__(self, **kwargs)
        ThinkingMixin.__init__(self, **kwargs)

        self._api_url: str = kwargs.pop('api_url', 'https://api.openai.com/v1')
        self._api_token: str = kwargs.pop('api_token', '')
        self._store_conversation: bool = kwargs.pop('store_conversation', False)
        self._image_quality: str = kwargs.pop('image_quality', 'auto')

        self._reasoning: bool = kwargs.pop('reasoning', False)
        self._reasoning_effort: str = kwargs.pop('reasoning_effort', 'low')
        self._reasoning_exclude_output: bool = kwargs.pop('reasoning_exclude_output', True)

        if not self._api_url:
            raise ValueError("no api_url")

        if not hasattr(self, '_api_type'):
            self._api_type = APITypes.OPENAI

        _logger.info('API set to %s.', self._api_type)

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
            self._load_model(model_repo)
        except requests.HTTPError as e:
            raise ValueError(self._normalize_error(e))
        except requests.ConnectionError:
            raise ValueError(f'api unavailable: {self._api_url}')

        _logger.info('Model set to %s.', self._current_model)

    def _load_model(self, model_repo: str):
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

    def unload_model(self):
        pass

    def offload_model(self):
        pass

    def conversation(self, image: DatasetImage, stream: bool = False, **kwargs):
        system_prompt, user_prompt = self._prompts_from_image(image, **kwargs)

        mime_type, encoded_image = self._encode_image(
            image,
            max_image_size=self._api_type.max_image_size,
            max_image_encoded_size=self._api_type.max_image_encoded_size,
            **kwargs
        )

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
            'model': self._current_model,
            'stream': stream,
            'store': self._store_conversation,
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
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}",
                                "detail": self._image_quality,
                            },
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
                "content": assistant_prefill,
                "is_prefill": True,
            })

        if stream:
            conversation['stream_options'] = {
                'include_usage': True,
            }

        if self._reasoning:
            conversation['reasoning_effort'] = self._reasoning_effort

        conversation.update(conversation_overrides)

        for key, value in list(conversation.items()):
            if value is None:
                conversation.pop(key)

        return conversation

    def _extract_assistant_prefill(self, conversation: dict):
        assistant_prefill = ''
        try:
            last_message = conversation['messages'][-1]
            if last_message.get('is_prefill', False):
                assistant_prefill = last_message['content']
                last_message.pop('is_prefill', None)
            else:
                assistant_prefill = ''

            assert isinstance(assistant_prefill, str)
        except Exception:
            _logger.debug('Failed to extract assistant prefill', exc_info=True)
            assistant_prefill = ''

        return assistant_prefill


    def _generate_stream_prediction_inner(self, image: DatasetImage, **kwargs):
        assert self._current_model, "model not loaded"

        # make sure stream is not set in kwargs
        kwargs.pop('stream', None)

        conversation = self.conversation(image, stream=True, **kwargs)
        assistant_prefill = self._extract_assistant_prefill(conversation)

        with self._session.post('chat/completions', stream=True, json=conversation) as conversation_resp:
            try:
                conversation_resp.raise_for_status()
            except:
                # NOTE: consume the stream so error can be parsed
                conversation_error = '\n'.join(conversation_resp.iter_lines(decode_unicode=True))
                conversation_error = conversation_error.strip()

                raise ErrorNormalizationMixin.GenerationError(conversation_error)

            if assistant_prefill:
                yield assistant_prefill

            converation_stopped = False

            is_thinking = False   # used to wrap the thoughts in <think>...</think>
            is_prediction = False # prevents the thoughts from being printed if the first thought is done

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
                    line_response = OpenAIChatCompletionChunkResponse(**line_json)

                    if line_response.object != CHAT_COMPLETION_CHUNK_OBJECT:
                        continue

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

                        if not is_prediction and (thought := choice.delta.reasoning or choice.delta.reasoning_content):
                            if not is_thinking:
                                yield self._reasoning_start_token
                                is_thinking = True

                            yield thought

                        content = choice.delta.content or choice.delta.refusal

                        if not content:
                            continue

                        if is_thinking:
                            yield self._reasoning_end_token
                            is_thinking = False

                        is_prediction = True

                        yield content
                        break # only retrieve first choice
                except pydantic.ValidationError:
                    _logger.error('Error: failed to process line: not a stream response: %s', line)
                    break
                except AssertionError as e:
                    _logger.error('Error: failed to process line: %s: %s', e, line)
                    break

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

        conversation = self.conversation(image, stream=False, **kwargs)
        assistant_prefill = self._extract_assistant_prefill(conversation)

        is_thinking = False   # used to wrap the thoughts in <think>...</think>

        with self._session.post('chat/completions', stream=False, json=conversation) as conversation_resp:
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
                conversation_response = OpenAIChatCompletionResponse(**conversation_json)
                assert conversation_response.object == CHAT_COMPLETION_OBJECT, f'api did not return a chat completion response'
            except AssertionError as e:
                _logger.debug('Failed to decode response to object: %s', conversation_resp.text)
                raise ValueError(str(e))
            except pydantic.ValidationError as e:
                _logger.debug('Failed to decode response to object: %s', conversation_resp.text)
                raise ValueError('api did not return a valid response')

            if conversation_response.usage and conversation_response.id != "SKIPPED":
                self._api_usage[conversation_response.id] = APIUsage(
                    response_tokens=conversation_response.usage.completion_tokens,
                    prompt_tokens=conversation_response.usage.prompt_tokens,
                    total_tokens=conversation_response.usage.total_tokens,
                    thoughts_tokens=0 if not conversation_response.usage.completion_tokens_details else conversation_response.usage.completion_tokens_details.reasoning_tokens,
                )

            thought_buffer = ''

            for choice in conversation_response.choices:
                if choice.finish_reason and choice.finish_reason != 'stop':
                    raise ValueError(self._normalize_error(conversation_response))

                thought_content = choice.message.reasoning or choice.message.reasoning_content
                if thought_content:
                    if not is_thinking:
                        thought_buffer += self._reasoning_start_token
                        is_thinking = True

                    thought_buffer += thought_content

                if is_thinking:
                    thought_buffer += self._reasoning_end_token
                    is_thinking = False

                content = choice.message.content or choice.message.refusal

                if not content:
                    continue

                if is_thinking:
                    thought_buffer += self._reasoning_end_token
                    is_thinking = False

                if assistant_prefill:
                    content = assistant_prefill + content

                return thought_buffer + content

            raise ValueError('api did not return text')


    def predict(self, image: DatasetImage, **kwargs):
        try:
            return self._handle_thinking(self._generate_prediction(image, **kwargs))
        except requests.HTTPError as e:
            raise ValueError(self._normalize_error(e))

    def predict_stream(self, image: DatasetImage, **kwargs):
        yield from self._handle_thinking_streaming(self._generate_stream_prediction(image, **kwargs))
