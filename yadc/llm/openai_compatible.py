"""``OpenAICompatibleLLMClient`` — chat/completions client for any OpenAI-style server.

This is the base for the per-server thin clients (``OpenAILLMClient``,
``OpenRouterLLMClient``, ``VllmLLMClient``, ``LlamacppLLMClient``,
``KoboldcppLLMClient``, ``OllamaLLMClient``). It owns the request
body-building, the SSE chunk decoding, reasoning translation, and the
connection-error wrapping shared by every ``/v1/chat/completions``-style
endpoint. Per-server tweaks (``max_tokens`` rename, bespoke
``load_model``, reasoning block shape) live in the subclasses.

:class:`yadc.captioners.api.APICaptioner` composes this client and adds
the image / Jinja / thinking glue. Backend selection (which subclass
matches a given URL) is :func:`yadc.llm.create_client`.
"""

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
import pydantic
from typing_extensions import override

from yadc.core import logging

from .base import APIUsage, BaseLLMClient
from .constants import DEFAULT_MODELS_CACHE_TTL_SECONDS
from .error_normalization import normalize_error
from .response_models import (
    OpenAIChatCompletionChunkResponse,
    OpenAIModelsResponse,
    _OpenAIReasoningDetailEncrypted,
    _OpenAIReasoningDetailSummary,
    _OpenAIReasoningDetailText,
)
from .types import Message, MessageStream, StreamChunk

_logger = logging.get_logger(__name__)

CHAT_COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Client for any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Subclasses (``OpenRouterLLMClient``, ``LlamacppLLMClient``, …) override
    :meth:`_customize_body` for per-server body tweaks (``max_tokens`` vs
    ``max_completion_tokens``, reasoning format, stop tokens) and may
    override :meth:`load_model` for bespoke model loading.
    """

    def __init__(self, **kwargs: Any):
        BaseLLMClient.__init__(self, **kwargs)

    @staticmethod
    def _is_reasoning_redacted(_text: str) -> bool:
        return False

    @override
    async def list_models(self, *, cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS) -> list[str]:
        """Fetch available model IDs from the OpenAI-compatible ``/models`` endpoint."""
        async with self._async_session.get("models", cache_ttl=cache_ttl) as model_resp:
            assert isinstance(model_resp, httpx.Response)
            model_resp.raise_for_status()

            model_resp_json = model_resp.json()
            assert isinstance(model_resp_json, dict), f"bad models response type; expected dict, got {type(model_resp_json)}"

            try:
                models = OpenAIModelsResponse.model_validate(model_resp_json)
            except pydantic.ValidationError as e:
                raise ValueError("failed to parse model list response") from e

            return [model.id for model in models.data]

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

        body = self._build_body(
            messages, resolved_model=resolved_model, max_tokens=max_tokens, reasoning=reasoning, reasoning_exclude=reasoning_exclude, opts=opts
        )

        try:
            async with self._async_session.capture_response(image_name=capture_name) as capture_ctx:
                handle = await self._async_session.open_stream("POST", "chat/completions", capture_ctx=capture_ctx, json=body)
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
        resolved_model: str,
        max_tokens: int,
        reasoning: Literal["low", "medium", "high"] | None,
        reasoning_exclude: bool,
        opts: dict[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": resolved_model,
            "stream": True,
            "max_completion_tokens": max_tokens,
            "messages": [self._message_to_wire(m) for m in messages],
            "stream_options": {"include_usage": True},
        }

        if reasoning is not None:
            body["reasoning_effort"] = reasoning

        # Transient key consumed by OpenRouter's `_customize_body` to build its
        # `reasoning.exclude` flag; popped after customization so it never leaks.
        body["_reasoning_exclude"] = reasoning_exclude

        body = self._customize_body(body)
        body.pop("_reasoning_exclude", None)

        body.update(opts)

        return {k: v for k, v in body.items() if v is not None}

    def _customize_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Per-server body hook. Default is a no-op; subclasses rename keys
        (``max_completion_tokens``→``max_tokens``), swap reasoning config
        shape, or add stop tokens."""
        return body

    def _message_to_wire(self, message: Message) -> dict[str, Any]:
        if isinstance(message.content, str):
            content: Any = message.content
        else:
            content = [part.model_dump(exclude_none=True) for part in message.content]

        wire: dict[str, Any] = {"role": message.role, "content": content}

        if message.reasoning:
            wire["reasoning_content"] = message.reasoning
        if message.reasoning_encrypted:
            wire["reasoning_details"] = list(message.reasoning_encrypted)

        return wire

    # ---- chunk decoding ----

    async def _decode_chunks(self, handle: Any) -> AsyncIterator[StreamChunk]:
        """Yield :class:`StreamChunk`\\ s from an open SSE response handle.

        Closes the handle on exhaustion or error. The close is idempotent, so
        ``MessageStream.cancel()`` can also close without double-closing.
        """
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
                    _logger.warning("Warning: failed to decode line: %s", line)
                    continue

                try:
                    assert isinstance(data, dict), "not a dict"
                    chunk_response = OpenAIChatCompletionChunkResponse.model_validate(data)

                    if chunk_response.object != CHAT_COMPLETION_CHUNK_OBJECT:
                        continue

                    if chunk_response.usage and chunk_response.id != "SKIPPED":
                        async with self._api_usage_lock:
                            self._api_usage[chunk_response.id] = APIUsage(
                                response_tokens=chunk_response.usage.completion_tokens,
                                prompt_tokens=chunk_response.usage.prompt_tokens,
                                total_tokens=chunk_response.usage.total_tokens,
                                thoughts_tokens=(
                                    0 if not chunk_response.usage.completion_tokens_details else chunk_response.usage.completion_tokens_details.reasoning_tokens
                                ),
                            )

                    if chunk_response.error:
                        raise ValueError(normalize_error(chunk_response))

                    for choice in chunk_response.choices:
                        if choice.finish_reason and choice.finish_reason != "stop":
                            raise ValueError(normalize_error(chunk_response))

                        delta = choice.delta

                        reasoning_text: str | None = None
                        summary: str | None = None
                        encrypted: list[dict[str, Any]] | None = None

                        if delta.reasoning_details:
                            ex_text, summary, encrypted = self._extract_reasoning_details(delta.reasoning_details)
                            if ex_text and reasoning_text is None:
                                reasoning_text = ex_text

                        if not is_prediction:
                            thought = delta.reasoning or delta.reasoning_content
                            if thought and reasoning_text is None:
                                reasoning_text = thought

                        content = delta.content or delta.refusal

                        if content is None and reasoning_text is None and summary is None and encrypted is None:
                            continue

                        if content is not None:
                            is_prediction = True

                        yield StreamChunk(text=content, reasoning=reasoning_text, reasoning_summary=summary, reasoning_encrypted=encrypted)
                        break
                except pydantic.ValidationError:
                    _logger.error("Error: failed to process line: not a stream response: %s", line)
                    break
                except AssertionError as e:
                    _logger.error("Error: failed to process line: %s: %s", e, line)
                    break
        except (httpx.RemoteProtocolError, httpx.ReadError) as e:
            _logger.warning("Stream connection closed unexpectedly: %s", e)
            raise ValueError("Connection closed unexpectedly by the server. The API may have shut down or become unreachable.") from e
        finally:
            await handle.aclose()

    def _extract_reasoning_details(self, details: list[Any]) -> tuple[str | None, str | None, list[dict[str, Any]] | None]:
        """Parse OpenAI ``reasoning.details`` into ``(text, summary, encrypted)``.

        Either one-or-more ``reasoning.text`` details (plain reasoning) or
        ``reasoning.summary`` + ``reasoning.encrypted`` details — never both.
        Mixed types raise.
        """
        has_text = any(isinstance(d, _OpenAIReasoningDetailText) for d in details)
        has_summary = any(isinstance(d, _OpenAIReasoningDetailSummary) for d in details)
        has_encrypted = any(isinstance(d, _OpenAIReasoningDetailEncrypted) for d in details)

        if has_text and (has_summary or has_encrypted):
            raise ValueError("received mixed reasoning detail types: reasoning.text cannot appear with reasoning.summary or reasoning.encrypted")

        reasoning_text: str | None = None
        summary: str | None = None
        encrypted: list[dict[str, Any]] | None = None

        for detail in details:
            if isinstance(detail, _OpenAIReasoningDetailText):
                if detail.text and not self._is_reasoning_redacted(detail.text):
                    reasoning_text = (reasoning_text or "") + detail.text
            elif isinstance(detail, _OpenAIReasoningDetailSummary):
                summary = (summary or "") + detail.summary
            elif isinstance(detail, _OpenAIReasoningDetailEncrypted):
                if encrypted is None:
                    encrypted = []
                encrypted.append(detail.model_dump())

        return reasoning_text, summary, encrypted
