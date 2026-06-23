"""``PromptGenerationService`` — thin DI wrapper around :func:`yadc.prompt_generation.stream_template_chunks`.

The API owns this service class so the DI system can auto-discover it
and the controller can inject it with a configured ``LoggingFactory``.
The actual env-load → validate → client-build → stream → cleanup
wiring lives in the neutral :mod:`yadc.prompt_generation` package; we
just forward ``generate()`` calls through and let the streaming core
do the work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from logging import Logger

from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.modules.service import Service
from yadc.llm import StreamChunk
from yadc.llm.async_session import AsyncSession
from yadc.llm.cache import HTTPResponseCache
from yadc.llm.response_logger import ResponseLogger
from yadc.prompt_generation import PromptGenerationRequest, stream_template_chunks


class PromptGenerationService(Service):
    """Generate a Jinja2 prompt template from an intent + few-shot examples.

    One-shot: instantiates a fresh LLM client per ``generate`` call and
    closes it when the stream is exhausted.
    """

    def __init__(self, logging: LoggingFactory) -> None:
        self._logger: Logger = logging.get_logger(__name__)

    async def generate(
        self,
        *,
        request: PromptGenerationRequest,
        password: str | None = None,
        cache: HTTPResponseCache | None = None,
        response_logger: ResponseLogger | None = None,
        async_session: AsyncSession | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield :class:`StreamChunk`\\s for the generated template body.

        Delegates to :func:`yadc.prompt_generation.stream_template_chunks`.
        ``PasswordRequiredError`` from ``cmd_envs.load_env`` propagates
        (the controller maps it to HTTP 403). ``cache`` and
        ``response_logger`` are forwarded to the LLM client; ``None``
        uses the client's defaults. ``async_session`` is exposed for tests.
        """
        self._logger.info(
            "Generating prompt template via env '%s' (focus=%s, examples=%d).",
            request.env,
            request.focus,
            len(request.examples),
        )
        async for chunk in stream_template_chunks(
            request,
            password=password,
            cache=cache,
            response_logger=response_logger,
            async_session=async_session,
        ):
            yield chunk


__all__ = ["PromptGenerationService"]
