"""``GeminiLLMClient`` — Google Gemini ``generateContent`` client.

Translates :class:`Message` lists to Gemini's ``system_instruction`` +
``contents`` + ``generationConfig`` (with optional ``thinkingConfig``)
shape, parses streaming responses (including ``thought: true`` reasoning
parts and ``usageMetadata``), and authenticates with ``x-goog-api-key``
instead of a Bearer token.

The captioning half (``APICaptioner``) composes this client.
"""

import json
from collections.abc import AsyncIterator
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import pydantic
from typing_extensions import override

from yadc.core import logging

from .base import APIUsage, BaseLLMClient
from .constants import DEFAULT_MODELS_CACHE_TTL_SECONDS
from .error_normalization import GenerationError, normalize_error
from .response_models import GeminiContentResponse, GeminiModel, GeminiModelsResponse
from .types import Message, MessageStream, StreamChunk, TextPart

_logger = logging.get_logger(__name__)


def _thinking_budget(effort: Literal["low", "medium", "high"]) -> int:
    return {"low": 512, "medium": 1024, "high": 2048}.get(effort, 512)


def _data_url_to_inline_data(url: str) -> dict[str, str] | None:
    """Parse a ``data:<mime>;base64,<data>`` URL into Gemini's inline_data shape."""
    parsed = urlparse(url)
    if parsed.scheme != "data":
        return None
    # data URLs look like "data:image/jpeg;base64,/9j/..."
    header, _, payload = url.partition(",")
    if ";base64" not in header:
        return None
    mime_type = header.removeprefix("data:").removesuffix(";base64")
    return {"mime_type": mime_type, "data": payload}


class GeminiLLMClient(BaseLLMClient):
    """Client for Google Gemini's ``generateContent`` API."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if not self._api_token:
            raise ValueError("no api_token")

        # Gemini authenticates with x-goog-api-key, not a Bearer token.
        self._async_session.headers.pop("Authorization", None)
        self._async_session.headers["x-goog-api-key"] = self._api_token

        self._is_thinking_model: bool = False

    @override
    async def list_models(self, *, cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS) -> list[str]:
        """Fetch generateContent-capable model IDs from Gemini's paginated ``/models`` endpoint."""
        return [model.name.removeprefix("models/") async for model in self._iter_generate_content_models(cache_ttl=cache_ttl)]

    async def _iter_generate_content_models(self, *, cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS):
        current_token: str | None = None
        next_token: str | None = None

        for _ in range(20):
            if next_token is not None and current_token == next_token:
                break

            path = f"models?pageToken={next_token}" if next_token is not None else "models"

            async with self._async_session.get(path, cache_ttl=cache_ttl) as models_resp:
                assert isinstance(models_resp, httpx.Response)
                models_resp.raise_for_status()

                models_resp_json = models_resp.json()
                assert isinstance(models_resp_json, dict), "bad model response"

                models = GeminiModelsResponse.model_validate(models_resp_json)

                for model in models.models:
                    if "generateContent" not in model.supportedGenerationMethods:
                        continue

                    model.name = model.name.removeprefix("models/")  # NOTE: some might return this prefix
                    yield model

                current_token = next_token
                next_token = models.nextPageToken

                if next_token is None:
                    break

    @override
    async def load_model(self, model_repo: str, **kwargs: Any) -> None:
        """Resolve the model and capture its thinking flag (``_is_thinking_model``).

        A direct ``GET models/{id}`` first; on miss, falls back to walking
        the paginated model list. The thinking flag gates the ``reasoning``
        kwarg in ``_build_body``.
        """
        try:
            model = await self._load_model(model_repo, **kwargs)
            self._model: str | None = model.name
            self._is_thinking_model = model.thinking

            _logger.info("Model set to %s.", self._model)
        except httpx.HTTPStatusError as e:
            raise ValueError(normalize_error(e)) from e
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ValueError(f"api unavailable: {self._api_url}") from e
        except AssertionError as e:
            raise ValueError(str(e)) from e

    async def _load_model(self, model_repo: str, **_kwargs: Any) -> GeminiModel:
        model_repo = model_repo.removeprefix("models/")

        # Fast path: fetch the model directly (common case; gets the thinking flag in one round-trip).
        async with self._async_session.get(f"models/{model_repo}", cache_ttl=DEFAULT_MODELS_CACHE_TTL_SECONDS) as model_resp:
            assert isinstance(model_resp, httpx.Response)
            if model_resp.status_code < 400:
                model = GeminiModel.model_validate(model_resp.json())

                if "generateContent" not in model.supportedGenerationMethods:
                    caps = ", ".join(model.supportedGenerationMethods)
                    raise ValueError(f"model {model_repo} does not have generative capabilities: available capabilities: {caps}")

                model.name = model.name.removeprefix("models/")  # NOTE: some might return this prefix
                return model

        # Discovery path: walk the paginated /models list to find the model
        # (and capture its thinking flag — list_models only collects names).
        matched_model: GeminiModel | None = None
        available_models: list[str] = []
        async for model in self._iter_generate_content_models():
            available_models.append(model.name)
            if model.name == model_repo:
                matched_model = model
                break

        if matched_model is None:
            if available_models:
                raise ValueError(f"model not found: {model_repo}; available models: {', '.join(available_models)}")

            raise ValueError(f"model not found: {model_repo}; no models available")

        return matched_model

    @override
    async def predict_next_message_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        model: str | None = None,
        reasoning: Literal["low", "medium", "high"] | None = None,
        reasoning_exclude: bool = False,
        capture_name: str = "llm_stream",
        **opts: Any,
    ) -> MessageStream:
        resolved_model = model or self._model
        if not resolved_model:
            raise ValueError("no model loaded; call load_model() first or pass model=...")
        resolved_model = resolved_model.removeprefix("models/")

        if reasoning is not None and not self._is_thinking_model:
            _logger.warning("Warning: requesting reasoning from a model that does not support thinking.")

        body = self._build_body(messages, max_tokens=max_tokens, reasoning=reasoning, reasoning_exclude=reasoning_exclude, opts=opts)

        path = f"models/{resolved_model}:streamGenerateContent?alt=sse"
        try:
            async with self._async_session.capture_response(image_name=capture_name) as capture_ctx:
                handle = await self._async_session.open_stream("POST", path, capture_ctx=capture_ctx, json=body)
        except (httpx.RemoteProtocolError, httpx.ReadError) as e:
            _logger.warning("Connection closed unexpectedly: %s", e)
            raise ValueError("Connection closed unexpectedly by the server. The API may have shut down or become unreachable.") from e

        try:
            handle.raise_for_status()
        except httpx.HTTPStatusError as e:
            await handle.aclose()
            raise ValueError(normalize_error(e)) from e

        message = Message(role="assistant", content="")
        chunks = self._decode_chunks(handle)
        return MessageStream(message=message, chunks=chunks, cancel_fn=handle.aclose)

    # ---- body building ----

    def _build_body(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        reasoning: Literal["low", "medium", "high"] | None,
        reasoning_exclude: bool,
        opts: dict[str, Any],
    ) -> dict[str, Any]:
        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []

        for m in messages:
            # ``developer`` is OpenAI's newer name for the system message; treat
            # it the same so the captioner's ``system_role`` override survives.
            if m.role in ("system", "developer"):
                for text in self._text_of(m):
                    system_parts.append({"text": text})
                continue

            role = "model" if m.role == "assistant" else "user"
            parts = self._message_to_parts(m)
            if m.reasoning:
                parts = [{"text": m.reasoning, "thought": True}, *parts]
            contents.append({"role": role, "parts": parts})

        generation_config: dict[str, Any] = {
            "maxOutputTokens": max_tokens,
            "responseModalities": ["TEXT"],
        }

        # Captioner passes image_quality as an opt; Gemini expresses it as
        # generationConfig.mediaResolution (OpenAI uses image_url `detail`).
        image_quality = opts.pop("image_quality", None)
        if image_quality == "low":
            generation_config["mediaResolution"] = "MEDIA_RESOLUTION_LOW"
        elif image_quality == "medium":
            generation_config["mediaResolution"] = "MEDIA_RESOLUTION_MEDIUM"
        elif image_quality == "high":
            generation_config["mediaResolution"] = "MEDIA_RESOLUTION_HIGH"

        if reasoning is not None and self._is_thinking_model:
            generation_config["thinkingConfig"] = {
                "includeThoughts": not reasoning_exclude,
                "thinkingBudget": _thinking_budget(reasoning),
            }

        # Advanced-settings nested overrides: merge generation_config into the
        # built config (e.g. temperature), and lift safety_settings to top-level.
        gen_overrides: dict[str, Any] | None = opts.pop("generation_config", None)
        if isinstance(gen_overrides, dict):
            generation_config.update(gen_overrides)
        safety_settings = opts.pop("safety_settings", opts.pop("safetySettings", None))

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        if system_parts:
            body["system_instruction"] = {"parts": system_parts}

        if safety_settings is not None:
            body["safetySettings"] = safety_settings

        body.update(opts)

        return {k: v for k, v in body.items() if v is not None}

    def _text_of(self, message: Message) -> list[str]:
        if isinstance(message.content, str):
            return [message.content]
        return [p.text for p in message.content if isinstance(p, TextPart)]

    def _message_to_parts(self, message: Message) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        content = message.content
        if isinstance(content, str):
            parts.append({"text": content})
            return parts
        for part in content:
            if isinstance(part, TextPart):
                parts.append({"text": part.text})
            else:
                inline = _data_url_to_inline_data(part.image_url.url)
                if inline is not None:
                    parts.append({"inline_data": inline})
        return parts

    # ---- chunk decoding ----

    async def _decode_chunks(self, handle: Any) -> AsyncIterator[StreamChunk]:
        is_prediction = False
        try:
            async for raw_line in handle.aiter_lines():
                line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith(":"):
                    continue

                line = line.removeprefix("data:").strip()

                if line == "[DONE]":
                    break

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    if line.startswith("{"):
                        raise GenerationError(line)
                    _logger.warning("Warning: failed to decode line: %s", line)
                    continue

                try:
                    assert isinstance(data, dict), "not a dict"
                    line_response = GeminiContentResponse.model_validate(data)

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
                            raise ValueError(normalize_error(line_response))

                        for part in candidate.content.parts:
                            if not part.text:
                                continue

                            if part.thought:
                                if is_prediction:
                                    continue
                                yield StreamChunk(reasoning=part.text)
                                continue

                            is_prediction = True
                            yield StreamChunk(text=part.text)
                            found_candidate = True
                            break
                except pydantic.ValidationError:
                    _logger.error("Error: failed to process line: not a stream response: %s", line)
                    break
                except AssertionError as e:
                    _logger.error("Error: failed to process line: %s: %s", e, line)
                    break
        except GenerationError as e:
            raise ValueError(normalize_error(e)) from e
        except (httpx.RemoteProtocolError, httpx.ReadError) as e:
            _logger.warning("Stream connection closed unexpectedly: %s", e)
            raise ValueError("Connection closed unexpectedly by the server. The API may have shut down or become unreachable.") from e
        finally:
            await handle.aclose()
