"""``GeminiCaptioner`` — Google Gemini ``generateContent`` backend.

Builds Gemini-specific request payloads (``system_instruction``,
``contents``, ``generationConfig``, ``safetySettings``,
optional ``thinkingConfig``) and parses responses including
``thought: true`` parts and ``usageMetadata`` for prompt/response/
thought token counts. Reasoning is enabled by translating
``reasoning_effort`` (``low``/``medium``/``high``) to a
``thinkingBudget`` (512/1024/2048 tokens). Like the OpenAI backend,
both ``predict`` and ``predict_stream`` catch
``httpx.RemoteProtocolError`` / ``httpx.ReadError`` and surface a
friendly error to the web UI.
"""

import copy
import json
from collections.abc import AsyncGenerator, AsyncIterator
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import pydantic
from typing_extensions import override

from yadc.core import DatasetImage, logging
from yadc.core.prediction import PredictionContext

from .base import BaseAPICaptioner
from .constants import DEFAULT_MODELS_CACHE_TTL_SECONDS
from .types import (
    GeminiContentResponse,
    GeminiModel,
    GeminiModelsResponse,
)
from .utils import ErrorNormalizationMixin, ThinkingMixin

_logger = logging.get_logger(__name__)


class APITypes(str, Enum):
    BASE = "base"

    @override
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
            case "low":
                return 512
            case "medium":
                return 1024
            case "high":
                return 2048

            # shouldn't happen, but we should have a default
            case _:
                return 512


class SafetySettings(pydantic.BaseModel):
    data: list["SafetySetting"] | None = None

    @pydantic.model_validator(mode="after")
    def validate_(self):
        existing_categories: set[str] = set()

        if self.data is None:
            return self

        for setting in self.data:
            if setting.category in existing_categories:
                raise ValueError(f"duplicate safetty setting: {setting.category}")

            existing_categories.add(setting.category)

        return self


class SafetySetting(pydantic.BaseModel):
    category: str
    threshold: str

    @pydantic.model_validator(mode="after")
    def validate_(self):
        try:
            assert self.category in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ), "invalid safety setting category: refer to documentation at https://ai.google.dev/api/generate-content#v1beta.HarmCategory"

            assert self.threshold in (
                "BLOCK_LOW_AND_ABOVE",
                "BLOCK_MEDIUM_AND_ABOVE",
                "BLOCK_ONLY_HIGH",
                "BLOCK_NONE",
                "OFF",
            ), f"invalid safety setting threshold for {self.category}: refer to documentation at https://ai.google.dev/api/generate-content#HarmBlockThreshold"
        except AssertionError as e:
            raise ValueError(e)

        return self


class APIUsage:
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    thoughts_tokens: int

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
        await captioner.load_model("gemini-pro-vision")
        caption = await captioner.predict(dataset_image)
    ```
    """

    _current_model: str | None = None
    _is_thinking_model: bool = False

    def __init__(self, **kwargs: Any):
        """
        Initializes the GeminiCaptioner with API and template configuration.

        Args:
            api_token (str): Google API key for authentication.

            **kwargs: Optional keyword arguments:
                - `api_url` (str): Base URL for the Gemini API endpoint.
                - `prompt_template` (str): The prompt template used for captioning. If none is provided, the default will be used.
                - `image_quality` (str): Quality setting for encoded images ('auto', 'low', 'high').
                - `reasoning` (bool): Enable internal chain-of-thought / extra reasoning behavior.
                - `reasoning_effort` (str, optional): Level of reasoning effort to request when `reasoning` is True ('low', 'medium', 'high').
                - `reasoning_exclude_output` (bool, optional): When True, exclude internal reasoning output from the caption.

        Raises:
            ValueError: If `api_token` is not provided.
        """

        BaseAPICaptioner.__init__(self, **kwargs)
        ThinkingMixin.__init__(self, **kwargs)

        self._api_url: str = kwargs.pop("api_url", "https://generativelanguage.googleapis.com/v1beta")
        self._api_token: str = kwargs.pop("api_token", "")
        self._image_quality: str = kwargs.pop("image_quality", "auto")

        self._reasoning: bool = kwargs.pop("reasoning", False)
        self._reasoning_effort: str = kwargs.pop("reasoning_effort", "low")
        self._reasoning_exclude_output: bool = kwargs.pop("reasoning_exclude_output", True)

        if not self._api_url:
            raise ValueError("no api_url")

        if not self._api_token:
            raise ValueError("no api_token")

        assert self._async_session is not None, "async session not available"
        self._async_session.headers = {"x-goog-api-key": self._api_token}

        self._api_usage: dict[str, APIUsage] = {}

    @override
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
            _logger.info("Used a total of %d tokens (prompt: %d, response: %d).", usage.total_tokens, usage.prompt_tokens, usage.response_tokens)
        else:
            _logger.info(
                "Used a total of %d tokens (prompt: %d, response: %d, reasoning: %d).",
                usage.total_tokens,
                usage.prompt_tokens,
                usage.response_tokens,
                usage.thoughts_tokens,
            )

    @override
    async def load_model(self, model_repo: str, **kwargs: Any) -> None:
        try:
            await self._load_model(model_repo, **kwargs)
        except httpx.HTTPStatusError as e:
            raise ValueError(self._normalize_error(e))
        except (httpx.ConnectError, httpx.TimeoutException):
            raise ValueError(f"api unavailable: {self._api_url}")
        except AssertionError as e:
            raise ValueError(str(e))

    async def _load_model(self, model_repo: str, **_kwargs: Any) -> None:
        assert self._async_session is not None, "async session not available"

        model_repo = model_repo.removeprefix("models/")

        # Fast path: fetch the model directly. This is the common case and
        # gives us the ``thinking`` flag in one round-trip.
        async with self._async_session.get(f"models/{model_repo}", cache_ttl=DEFAULT_MODELS_CACHE_TTL_SECONDS) as model_resp:
            assert isinstance(model_resp, httpx.Response)
            if model_resp.status_code < 400:
                model_resp_json = model_resp.json()
                model = GeminiModel.model_validate(model_resp_json)

                if "generateContent" not in model.supportedGenerationMethods:
                    caps = ", ".join(model.supportedGenerationMethods)
                    raise ValueError(f"model {model_repo} does not have generative capabilities: available capabilities: {caps}")

                self._is_thinking_model = model.thinking
                self._current_model = model.name.removeprefix("models/")
                _logger.info("Model set to %s.", self._current_model)
                self._warn_if_reasoning_unsupported()

                return

        # Discovery path: walk the paginated ``/models`` list and find the
        # requested one. We iterate the raw ``GeminiModel`` objects so we
        # can capture the ``thinking`` flag of the matched model — that
        # flag drives ``thinkingConfig`` in :meth:`conversation` and would
        # be wrong if we just checked unprefixed name membership.
        #
        # Capability filtering (``generateContent``) is handled implicitly
        # by :meth:`_iter_generate_content_models` — unlike the fast path
        # above, we don't need a separate ``generateContent`` check here.
        matched_thinking: bool | None = None
        available_models: list[str] = []
        async for model in self._iter_generate_content_models():
            model_id = model.name.removeprefix("models/")
            available_models.append(model_id)
            if model_id == model_repo and matched_thinking is None:
                matched_thinking = model.thinking

        if matched_thinking is None:
            if available_models:
                raise ValueError(f"model not found: {model_repo}; available models: {', '.join(available_models)}")

            raise ValueError(f"model not found: {model_repo}; no models available")

        self._is_thinking_model = matched_thinking
        self._current_model = model_repo
        _logger.info("Model set to %s.", self._current_model)
        self._warn_if_reasoning_unsupported()

    def _warn_if_reasoning_unsupported(self) -> None:
        """Log a warning if the user enabled reasoning but the loaded
        model does not support thinking.

        Called by both the fast and discovery paths of :meth:`_load_model`
        after ``_is_thinking_model`` has been set, so the warning fires
        regardless of which path loaded the model.
        """
        if self._reasoning and not self._is_thinking_model:
            _logger.warning("Warning: selected a model without reasoning capabilities, but reasoning is enabled.")

    @override
    async def list_models(self, cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS) -> list[str]:
        """Fetch the list of model IDs from Gemini's paginated ``/models`` endpoint.

        Filters to models that support ``generateContent`` and strips the
        ``models/`` prefix from each name. Paginates via ``nextPageToken``
        until exhausted.
        """
        return [model.name.removeprefix("models/") async for model in self._iter_generate_content_models(cache_ttl=cache_ttl)]

    async def _iter_generate_content_models(self, *, cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS) -> AsyncIterator[GeminiModel]:
        """Yield ``GeminiModel`` objects that support ``generateContent``.

        Shared by :meth:`list_models` (which only needs the names) and
        :meth:`_load_model` (which needs the ``thinking`` flag of the
        matched model). The raw objects are yielded so each caller picks
        the fields it cares about.
        """
        assert self._async_session is not None, "async session not available"

        next_token: str | None = None

        while True:
            if next_token is not None:
                models_url_path = f"models?pageToken={next_token}"
            else:
                models_url_path = "models"

            async with self._async_session.get(models_url_path, cache_ttl=cache_ttl) as models_resp:
                assert isinstance(models_resp, httpx.Response)
                models_resp.raise_for_status()

                models_resp_json = models_resp.json()
                assert isinstance(models_resp_json, dict), "bad model response"

                models = GeminiModelsResponse.model_validate(models_resp_json)

                for model in models.models:
                    if "generateContent" not in model.supportedGenerationMethods:
                        continue
                    yield model

                next_token = models.nextPageToken

                if next_token is None:
                    break

    @override
    def unload_model(self) -> None:
        pass

    @override
    def offload_model(self) -> None:
        pass

    def conversation(self, image: DatasetImage, **kwargs: Any) -> dict[str, Any]:
        system_prompt, user_prompt = self.prompts_from_image(image, **kwargs)

        mime_type, encoded_image = self._encode_image(
            image, max_image_size=APITypes.BASE.max_image_size, max_image_encoded_size=APITypes.BASE.max_image_encoded_size, **kwargs
        )

        try:
            conversation_overrides: dict[str, Any] = kwargs.pop("conversation_overrides", {})
            assert isinstance(conversation_overrides, dict), (
                f"bad value for conversation_overrides/advanced settings; expected a dict, got: {type(conversation_overrides)}"
            )

            conversation_overrides = copy.deepcopy(conversation_overrides)

            # just make sure this is not overridden
            conversation_overrides.pop("contents", None)

            system_role = conversation_overrides.pop("system_role", None) or "system"
            assert isinstance(system_role, str), f"bad value for conversation_overrides/advanced settings system_role; expected a str, got: {type(system_role)}"

            user_role = conversation_overrides.pop("user_role", None) or "user"
            assert isinstance(user_role, str), f"bad value for conversation_overrides/advanced settings user_role; expected a str, got: {type(user_role)}"

            assistant_role = conversation_overrides.pop("assistant_role", None) or "model"
            assert isinstance(assistant_role, str), (
                f"bad value for conversation_overrides/advanced settings assistant_role; expected a str, got: {type(assistant_role)}"
            )

            assistant_prefill = conversation_overrides.pop("assistant_prefill", "")
            assert isinstance(assistant_prefill, str), (
                f"bad value for conversation_overrides/advanced settings assistant_prefill; expected a str, got: {type(assistant_prefill)}"
            )

            try:
                safety_settings_overrides = SafetySettings(data=conversation_overrides.get("safety_settings", None))
            except pydantic.ValidationError as e:
                raise AssertionError(f"bad value for conversation_overrides/advanced settings safety settings: {e}")

            generation_config_overrides: dict[str, Any] = conversation_overrides.get("generation_config", {})
            assert isinstance(generation_config_overrides, dict), (
                f"bad value for conversation_overrides/advanced settings generation_config; expected a dict, got: {type(generation_config_overrides)}"
            )

            extra_messages: list[Any] | None = kwargs.pop("extra_messages", None)
            assert extra_messages is None or isinstance(extra_messages, list), f"bad value for extra_messages; expected a list, got: {type(extra_messages)}"
        except AssertionError as e:
            raise ValueError(e)

        max_tokens: int = kwargs.pop("max_new_tokens", 512)

        generation_config: dict[str, Any] = {
            "maxOutputTokens": max_tokens,
            "responseModalities": ["TEXT"],
        }

        if self._image_quality == "low":
            generation_config["mediaResolution"] = "MEDIA_RESOLUTION_LOW"
        elif self._image_quality == "medium":
            generation_config["mediaResolution"] = "MEDIA_RESOLUTION_MEDIUM"
        elif self._image_quality == "high":
            generation_config["mediaResolution"] = "MEDIA_RESOLUTION_HIGH"

        conversation: dict[str, Any] = {
            "system_instruction": {
                "parts": [
                    {"text": system_prompt},
                ],
            },
            "contents": [
                {
                    "role": user_role,
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": encoded_image}},
                        {"text": user_prompt},
                    ],
                }
            ],
            "generationConfig": generation_config,
        }

        if assistant_prefill:
            conversation["contents"].append(
                {
                    "role": assistant_role,
                    "parts": [{"text": assistant_prefill}],
                    "is_prefill": True,
                }
            )

        if extra_messages:
            from yadc.core.captioner import ROLE_ASSISTANT, ROLE_USER, ReplyRound

            for msg in extra_messages:
                assert isinstance(msg, ReplyRound), f"extra_messages must be ReplyRound instances, got {type(msg)}"
                assert msg.role in (ROLE_USER, ROLE_ASSISTANT), f"extra_messages role must be ROLE_USER or ROLE_ASSISTANT, got {msg.role!r}"

                role = msg.role
                if role == ROLE_ASSISTANT:
                    role = assistant_role
                elif role == ROLE_USER:
                    role = user_role

                parts: list[dict[str, Any]] = []
                if msg.reasoning:
                    parts.append({"text": msg.reasoning, "thought": True})
                parts.append({"text": msg.content})

                conversation["contents"].append(
                    {
                        "role": role,
                        "parts": parts,
                    }
                )

        if self._is_thinking_model and self._reasoning:
            conversation["generationConfig"]["thinkingConfig"] = {
                "includeThoughts": not self._reasoning_exclude_output,
                "thinkingBudget": APITypes.BASE.get_thinking_budget(self._reasoning_effort),
            }

        if safety_settings_overrides.data is not None:
            conversation["safetySettings"] = safety_settings_overrides.model_dump()["data"]

        if generation_config_overrides:
            generation_config.update(generation_config_overrides)

        return conversation

    def _extract_assistant_prefill(self, conversation: dict[str, Any]) -> str:
        try:
            last_message = conversation["contents"][-1]
            if last_message.get("is_prefill", False):
                assistant_prefill = last_message["parts"][0]["text"]
                last_message.pop("is_prefill", None)
            else:
                assistant_prefill = ""

            assert isinstance(assistant_prefill, str)
        except Exception:
            _logger.debug("Failed to extract assistant prefill", exc_info=True)
            assistant_prefill = ""

        return assistant_prefill

    async def _generate_prediction(self, image: DatasetImage, prediction_context: PredictionContext | None = None, **kwargs: Any):
        assert self._current_model, "model not loaded"
        assert self._async_session is not None, "async session not available"

        kwargs.pop("stream", None)

        conversation = self.conversation(image, **kwargs)
        assistant_prefill = self._extract_assistant_prefill(conversation)

        is_thinking = False

        async with self._async_session.capture_response(image_name=Path(image.path).stem) as _ctx:
            async with self._async_session.post(
                f"models/{self._current_model}:generateContent",
                stream=False,
                json=conversation,
                capture_ctx=_ctx,
            ) as conversation_resp:
                assert isinstance(conversation_resp, httpx.Response)
                conversation_resp.raise_for_status()

                try:
                    conversation_json = json.loads(await conversation_resp.aread())
                    assert isinstance(conversation_json, dict), "api did not return valid json"
                except AssertionError as e:
                    _logger.debug("Failed to decode response to json: %s", await conversation_resp.aread())
                    raise ValueError(str(e))
                except json.JSONDecodeError:
                    _logger.debug("Failed to decode response to json: %s", await conversation_resp.aread())
                    raise ValueError("api did not return json")

                try:
                    conversation_response = GeminiContentResponse.model_validate(conversation_json)
                except AssertionError as e:
                    _logger.debug("Failed to decode response to object: %s", await conversation_resp.aread())
                    raise ValueError(str(e))
                except pydantic.ValidationError:
                    _logger.debug("Failed to decode response to object: %s", await conversation_resp.aread())
                    raise ValueError("api did not return a valid response")

                if conversation_response.usageMetadata and conversation_response.responseId != "SKIPPED":
                    async with self._api_usage_lock:
                        self._api_usage[conversation_response.responseId] = APIUsage(
                            response_tokens=conversation_response.usageMetadata.candidatesTokenCount,
                            prompt_tokens=conversation_response.usageMetadata.promptTokenCount,
                            total_tokens=conversation_response.usageMetadata.totalTokenCount,
                            thoughts_tokens=conversation_response.usageMetadata.thoughtsTokenCount,
                        )

                thought_buffer = ""

                for candidate in conversation_response.candidates:
                    if candidate.finishReason and candidate.finishReason != "STOP":
                        raise ValueError(self._normalize_error(conversation_response))

                    for part in candidate.content.parts:
                        if text := part.text:
                            if not part.thought:
                                continue

                            if prediction_context is not None:
                                prediction_context.reasoning = (prediction_context.reasoning or "") + text

                            if not is_thinking:
                                thought_buffer += self._reasoning_start_token
                                is_thinking = True

                            thought_buffer += text

                    if is_thinking:
                        thought_buffer += self._reasoning_end_token
                        is_thinking = False

                    for part in candidate.content.parts:
                        if text := part.text:
                            if part.thought:
                                continue

                            if assistant_prefill:
                                text = assistant_prefill + text

                            return thought_buffer + text

                raise ValueError("api did not return text")

    async def _agenerate_stream_prediction_inner(self, image: DatasetImage, prediction_context: PredictionContext | None = None, **kwargs: Any):
        assert self._current_model, "model not loaded"
        assert self._async_session is not None, "async session not available"

        conversation = self.conversation(image, **kwargs)
        assistant_prefill = self._extract_assistant_prefill(conversation)

        async with self._async_session.capture_response(image_name=Path(image.path).stem) as _ctx:
            async with self._async_session.post(
                f"models/{self._current_model}:streamGenerateContent?alt=sse",
                stream=True,
                json=conversation,
                capture_ctx=_ctx,
            ) as conversation_resp:
                try:
                    conversation_resp.raise_for_status()
                except Exception:
                    lines: list[str] = []
                    async for raw_line in conversation_resp.aiter_lines():
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        lines.append(line)
                    conversation_error = "\n".join(lines)
                    conversation_error = conversation_error.strip()

                    raise ErrorNormalizationMixin.GenerationError(conversation_error)

                if assistant_prefill:
                    yield assistant_prefill

                conversation_error = ""
                conversation_stopped = False

                is_thinking = False
                is_prediction = False

                async for raw_line in conversation_resp.aiter_lines():
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line or conversation_stopped:
                        continue

                    try:
                        if line.startswith(":"):
                            continue

                        line = line.removeprefix("data:").strip()

                        if line == "[DONE]":
                            conversation_stopped = True
                            continue

                        line_json = json.loads(line)
                    except json.JSONDecodeError:
                        if line.startswith("{"):
                            conversation_error = line + "\n"
                            break

                        _logger.warning("Warning: failed to decode line: %s", line)
                        continue

                    try:
                        assert isinstance(line_json, dict), "not a dict"
                        line_response = GeminiContentResponse.model_validate(line_json)

                        if line_response.usageMetadata and line_response.responseId != "SKIPPED":
                            async with self._api_usage_lock:
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

                            if candidate.finishReason and candidate.finishReason != "STOP":
                                raise ValueError(self._normalize_error(line_response))

                            for part in candidate.content.parts:
                                if text := part.text:
                                    if part.thought:
                                        if is_prediction:
                                            continue

                                        if prediction_context is not None:
                                            prediction_context.reasoning = (prediction_context.reasoning or "") + text

                                        if not is_thinking:
                                            yield self._reasoning_start_token
                                            is_thinking = True

                                        yield text

                                        continue

                                    if is_thinking:
                                        yield self._reasoning_end_token
                                        is_thinking = False

                                    is_prediction = True

                                    yield text

                                    found_candidate = True
                                    break
                    except pydantic.ValidationError:
                        _logger.error("Error: failed to process line: not a stream response: %s", line)
                        break
                    except AssertionError as e:
                        _logger.error("Error: failed to process line: %s: %s", e, line)
                        break

                if conversation_error:
                    error_lines: list[str] = []
                    async for raw_line in conversation_resp.aiter_lines():
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        error_lines.append(line)
                    conversation_error += "\n".join(error_lines)
                    conversation_error = conversation_error.strip()

                    raise ErrorNormalizationMixin.GenerationError(conversation_error)

    async def _generate_stream_prediction(self, image: DatasetImage, **kwargs: Any) -> AsyncGenerator[str, None]:
        try:
            async for token in self._agenerate_stream_prediction_inner(image, **kwargs):
                yield token
            return
        except httpx.HTTPStatusError as e:
            raise ValueError(self._normalize_error(e))
        except ErrorNormalizationMixin.GenerationError as e:
            raise ValueError(self._normalize_error(e))
        except (httpx.RemoteProtocolError, httpx.ReadError) as e:
            _logger.warning("Stream connection closed unexpectedly: %s", e)
            raise ValueError("Connection closed unexpectedly by the server. The API may have shut down or become unreachable.") from e

    @override
    async def predict(self, image: DatasetImage, **kwargs: Any) -> str:
        self._before_predict(kwargs)
        try:
            return self._handle_thinking(await self._generate_prediction(image, **kwargs))
        except httpx.HTTPStatusError as e:
            raise ValueError(self._normalize_error(e))
        except (httpx.RemoteProtocolError, httpx.ReadError) as e:
            _logger.warning("Connection closed unexpectedly: %s", e)
            raise ValueError("Connection closed unexpectedly by the server. The API may have shut down or become unreachable.") from e

    @override
    async def predict_stream(self, image: DatasetImage, **kwargs: Any) -> AsyncGenerator[str, None]:
        self._before_predict(kwargs)
        async for token in self._handle_thinking_streaming_async(self._generate_stream_prediction(image, **kwargs)):
            yield token
