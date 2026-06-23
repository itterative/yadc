"""Streaming core for prompt-template generation.

Single source of truth for the env-load → validate → client-build →
stream → cleanup wiring. Both the CLI
(:func:`yadc.cmd.prompts.generate`) and the API service
(:class:`yadc.api.services.prompt_generation.PromptGenerationService`)
call this, so changes to env handling, validation, or client lifecycle
happen in exactly one place.

Raises:

- :class:`yadc.cmd.envs.keystorage_password.PasswordRequiredError` —
  propagates from ``cmd_envs.load_env`` untouched (callers map it to
  TTY prompt for CLI, 403 for API).
- :class:`PromptGenerationConfigError` — env is missing
  ``api_url`` / ``api_model_name``.
- Any exception from the underlying LLM client (network, model
  errors) propagates; ``client.aclose()`` is always called from the
  ``finally`` block.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from yadc.cmd import envs as cmd_envs
from yadc.llm import MessageStream, StreamChunk, create_client
from yadc.llm.async_session import AsyncSession
from yadc.llm.cache import HTTPResponseCache
from yadc.llm.response_logger import ResponseLogger

from .messages import (
    _GENERATE_SYSTEM_PROMPT,
    _REFINE_SYSTEM_PROMPT,
    _build_messages,
)
from .models import PromptGenerationConfigError, PromptGenerationRequest


async def stream_template_chunks(
    request: PromptGenerationRequest,
    *,
    password: str | None = None,
    cache: HTTPResponseCache | None = None,
    response_logger: ResponseLogger | None = None,
    async_session: AsyncSession | None = None,
) -> AsyncIterator[StreamChunk]:
    """Yield :class:`StreamChunk`\\s for the generated template body.

    ``PasswordRequiredError`` from ``cmd_envs.load_env`` propagates
    untouched. ``cache`` and ``response_logger`` are forwarded to the
    LLM client; ``None`` uses the client's defaults. ``async_session``
    is exposed for tests that want to inject a mock session.
    """
    config = cmd_envs.load_env(request.env, password=password)

    if not config.api.url:
        raise PromptGenerationConfigError(f"environment '{request.env}' has no api_url configured")

    model_name = request.api_model_name or config.api.model_name
    if not model_name:
        raise PromptGenerationConfigError(f"environment '{request.env}' has no api_model_name configured; pass api_model_name explicitly")

    if async_session is None:
        headers: dict[str, str] = {}
        if config.api.token:
            headers["Authorization"] = f"Bearer {config.api.token}"
        async_session = AsyncSession(config.api.url, headers=headers, cache=cache, response_logger=response_logger)

    client = await create_client(
        api_url=config.api.url,
        api_token=config.api.token,
        cache=cache,
        response_logger=response_logger,
        async_session=async_session,
    )

    try:
        await client.load_model(model_name)
        messages = _build_messages(
            request,
            system_prompt=_GENERATE_SYSTEM_PROMPT,
            refine_system_prompt=_REFINE_SYSTEM_PROMPT,
            image_quality=request.image_quality,
        )
        stream: MessageStream = await client.predict_next_message_stream(
            messages,
            reasoning=None,
            max_tokens=request.max_tokens,
            # Gemini reads ``image_quality`` from opts and maps it to
            # ``mediaResolution``; OpenAI-compatible clients pick it up
            # from the message's ``image_url.detail`` instead. Passing
            # it here too covers both backends (mirrors the captioner).
            image_quality=request.image_quality,
        )
        async for chunk in stream:
            yield chunk
    finally:
        # ``client.aclose`` also closes the underlying ``async_session``
        # (which is owned by the client unless injected). httpx
        # ``aclose`` is idempotent, so re-closing is safe either way.
        await client.aclose()


__all__ = ["stream_template_chunks"]
