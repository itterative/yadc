"""``normalize_error()`` — turns an error into a stable, captioned string.

Async because ``httpx.HTTPStatusError`` raised from a streaming response
carries a body that hasn't been read yet — accessing ``.text`` would
raise ``ResponseNotRead``. ``normalize_error`` drains the body via
``await response.aread()`` before reading the text. Non-streaming and
``requests.HTTPError`` paths (sync, body always pre-read) are no-ops
for the read.

Called from ``except`` blocks by both the API captioners
(``yadc/captioners/api/``) and the LLM client layer (``yadc/llm/``).
Accepts ``httpx.HTTPStatusError`` / ``requests.HTTPError``, the
``GenerationError`` exception, or a parsed Pydantic response object
(``OpenAIChatCompletionResponse`` / ``OpenAIChatCompletionChunkResponse``
/ ``GeminiContentResponse``), and returns a human-readable ``str`` that
the web UI surfaces verbatim — so it must not contain stack-trace noise.

Parsing order for error-bearing payloads:

1. OpenAI error JSON (``OpenAIErrorResponse``); a ``metadata.moderation``
   block is treated as an OpenRouter moderation error (join the
   ``reasons`` list).
2. ``GeminiErrorResponse`` (wraps the error code + status in the message).
3. HTTP status code map (400/401/402/404/408/429/502/503 → canned text).

The parsed response objects instead report *why generation stopped*
(``finish_reason`` / ``promptFeedback`` / moderation), since some backends
signal errors that way rather than raising.

``GenerationError`` is a separate exception that callers raise after a
malformed/unexpected line; ``normalize_error`` parses its string payload
with the same OpenAI/Gemini logic.
"""

import json
from typing import Any

import httpx
import requests

from yadc.core import logging

from .response_models import (
    GeminiContentResponse,
    GeminiErrorResponse,
    OpenAIChatCompletionChunkResponse,
    OpenAIChatCompletionResponse,
    OpenAIErrorResponse,
    OpenRouterModerationError,
)

_logger = logging.get_logger(__name__)


class _ParsedError:
    source: str
    code: int | str
    message: str

    def __init__(self, error_source: str, error_code: int | str, error_message: str):
        self.source = error_source
        self.code = error_code
        self.message = error_message


class GenerationError(Exception):
    def __init__(self, error: str) -> None:
        super().__init__(error)


async def normalize_error(error: Any) -> str:
    def _try_parse_moderation(moderation: dict[str, Any]) -> _ParsedError | None:
        try:
            moderation_error = OpenRouterModerationError.model_validate(moderation)
            return _ParsedError("moderation", 400, "; ".join(moderation_error.reasons))
        except Exception:
            pass

        return None

    def _try_parse_error_json(error_source: str, error_text: str):
        try:
            assert error_text

            error_json = json.loads(error_text)
            assert isinstance(error_json, dict)
        except Exception:
            return None

        try:
            error_response = OpenAIErrorResponse.model_validate(error_json).error

            error_code = error_response.code
            error_message = error_response.message

            if error_response.metadata and (error_parsed := _try_parse_moderation(error_response.metadata)):
                return error_parsed

            return _ParsedError(error_source, error_code, error_message)
        except Exception:
            pass

        try:
            error_response = GeminiErrorResponse.model_validate(error_json).error

            error_code = error_response.code
            error_message = error_response.message

            return _ParsedError(error_source, error_code, f"({error_response.status}) {error_message}")
        except Exception:
            pass

        return None

    error_source = "app"
    error_code = -1
    error_message = ""

    if isinstance(error, (requests.HTTPError, httpx.HTTPStatusError)):
        # ``httpx.HTTPStatusError`` raised from a streaming response
        # carries a response whose body hasn't been consumed — accessing
        # ``.text`` raises ``ResponseNotRead``. Drain the body first;
        # ``aread()`` is a no-op for non-streaming / already-read responses.
        if isinstance(error, httpx.HTTPStatusError):
            try:
                await error.response.aread()
            except Exception as read_exc:
                _logger.warning("Warning: failed to read error response body: %s", read_exc)

        try:
            error_text = error.response.text
        except httpx.ResponseNotRead:
            error_text = ""

        _logger.debug("HTTP error %d: headers %s", error.response.status_code, error.response.headers)
        _logger.debug("HTTP error %d: %s", error.response.status_code, error_text)

        error_source = "http"
        error_code = error.response.status_code
        error_message = ""

        if error_parsed := _try_parse_error_json(error_source, error_text):
            return f"api returned an error ({error_parsed.source} {error_parsed.code}): {error_parsed.message}"

        _logger.warning("Warning: failed to process http error: %d", error.response.status_code)
    elif isinstance(error, GenerationError):
        _logger.debug("Generation error: %s", error)

        error_source = "generation"
        error_code = 500

        if error_parsed := _try_parse_error_json(error_source, str(error)):
            return f"api returned an error ({error_parsed.source} {error_parsed.code}): {error_parsed.message}"

        _logger.warning("Warning: failed to process generation error")
    elif isinstance(error, OpenAIChatCompletionResponse):
        for choice in error.choices:
            if not choice.finish_reason:
                continue

            return f"api stopped generating: reason: {choice.finish_reason}"

        _logger.warning("Warning: failed to process openai response")
    elif isinstance(error, OpenAIChatCompletionChunkResponse):
        error_response = error.error

        if error_response is not None:
            if error_response.metadata and (error_parsed := _try_parse_moderation(error_response.metadata)):
                return f"api stopped generating: reason: moderation: {error_parsed.message}"

        for choice in error.choices:
            if not choice.finish_reason:
                continue

            return f"api stopped generating: reason: {choice.finish_reason}"

        _logger.warning("Warning: failed to process openai streaming response")
    elif isinstance(error, GeminiContentResponse):
        if error.promptFeedback:
            return f"api stopped generating: reason: {error.promptFeedback.blockReason}: {'; '.join(error.promptFeedback.safetyRatings)}"

        for candidate in error.candidates:
            if not candidate.finishReason:
                continue

            return f"api stopped generating: reason: {candidate.finishReason}"

        _logger.warning("Warning: failed to process gemini streaming response")
    else:
        _logger.debug("Unhandled error %s: %s", type(error), error)

        return f"unknown error: {type(error)}"

    # some defaults if response cannot be processed
    if not error_message:
        match (error_source, error_code):
            case ("http", 400):
                error_message = "request could not be completed"
            case ("http", 401):
                error_message = "authentication failure"
            case ("http", 402):
                error_message = "not enough api credits; payment needed"
            case ("http", 404):
                error_message = "model not found"
            case ("http", 408):
                error_message = "timeout"
            case ("http", 429):
                error_message = "overloaded"
            case ("http", 502):
                error_message = "unavailable"
            case ("http", 503):
                error_message = "unavailable"
            case ("app", _):
                error_message = "unknown error"
            case _:
                error_message = "unknown error"

    return f"api returned an error ({error_source} {error_code}): {error_message}"
