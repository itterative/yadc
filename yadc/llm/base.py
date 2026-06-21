"""``BaseLLMClient`` — shared HTTP infrastructure and the abstract client surface.

Each concrete client (``OpenAICompatibleLLMClient``, ``GeminiLLMClient``,
and the per-server thin subclasses) extends this. The base owns the
:class:`AsyncSession` (retries, UA, base URL, timeouts), the optional
file-based :class:`HTTPResponseCache`, and the optional
:class:`ResponseLogger`, plus the per-instance model + token-usage
bookkeeping shared by every backend.

``predict_next_message_stream`` is the one abstract prediction method —
stream-only. Callers that don't care about chunks just
``await stream.collect()``. See :class:`yadc.llm.types.MessageStream`.
"""

import abc
import asyncio
from typing import Any

import httpx

from yadc.core import logging

from .async_session import AsyncSession
from .cache import HTTPResponseCache
from .constants import DEFAULT_MODELS_CACHE_TTL_SECONDS
from .response_logger import ResponseLogger
from .types import Message, MessageStream

_logger = logging.get_logger(__name__)


class APIUsage:
    """Accumulated token usage for a single API response."""

    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    thoughts_tokens: int

    def __init__(self, prompt_tokens: int, response_tokens: int, total_tokens: int, thoughts_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.total_tokens = total_tokens
        self.thoughts_tokens = thoughts_tokens


class BaseLLMClient(abc.ABC):
    """Abstract base for all LLM backend clients.

        Construct with ``api_url`` (required), ``api_token`` (optional), and
        optionally wire ``cache``, ``response_logger``, and an injectable
    ``async_session`` (primarily for tests). When no session is supplied the
        base constructs one from the URL + token + timeouts.
    """

    def __init__(
        self,
        *,
        api_url: str,
        api_token: str = "",
        cache: HTTPResponseCache | None = None,
        response_logger: ResponseLogger | None = None,
        async_session: AsyncSession | None = None,
        connect_timeout: float = 30.0,
        read_timeout: float | None = None,
        write_timeout: float = 30.0,
        pool_timeout: float = 30.0,
        warnings: bool = True,
        **_kwargs: Any,
    ):
        if not api_url:
            raise ValueError("no api_url")

        self._api_url: str = api_url
        self._api_token: str = api_token
        self._cache: HTTPResponseCache | None = cache
        self._response_logger: ResponseLogger | None = response_logger
        self._model: str | None = None

        self._api_usage: dict[str, APIUsage] = {}

        # Guards the per-response `_api_usage` dict so concurrent predictions
        # don't race on the end-of-response mutation. Held only for the brief
        # dict write, so it doesn't serialise predictions.
        self._api_usage_lock: asyncio.Lock = asyncio.Lock()

        if async_session is not None:
            self._async_session: AsyncSession = async_session
        else:
            headers: dict[str, str] = {}
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"
            elif warnings:
                _logger.warning("Warning: no api_token is set, requests will fail if api uses authentication")

            self._async_session = AsyncSession(
                api_url,
                headers=headers,
                cache=cache,
                response_logger=response_logger,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                write_timeout=write_timeout,
                pool_timeout=pool_timeout,
            )

    @property
    def api_url(self) -> str:
        return self._api_url

    @property
    def model(self) -> str | None:
        return self._model

    @abc.abstractmethod
    async def predict_next_message_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        model: str | None = None,
        reasoning: Any = None,
        reasoning_exclude: bool = False,
        capture_name: str = "llm_stream",
        **opts: Any,
    ) -> MessageStream:
        """POST the request, open the streaming response, return a ``MessageStream``.

        Awaiting this performs the backend POST and obtains the open
        streaming response — preflight / connection errors (including the
        friendly ``ValueError("Connection closed unexpectedly…")`` wrapper
        for ``httpx.RemoteProtocolError`` / ``httpx.ReadError``) surface
        here. The returned ``MessageStream`` wraps the already-open
        response, so cancellation is a concrete close operation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def list_models(self, *, cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS) -> list[str]:
        """Return the model IDs available on this backend."""
        raise NotImplementedError

    async def load_model(self, model_repo: str, **_kwargs: Any) -> None:
        """Validate the model exists. Local backends override to load into VRAM.

        No-op if ``model_repo`` is already loaded. Default implementation
        lists models and checks ``model_repo`` is present, raising a
        ``ValueError`` with a short list of alternatives otherwise.
        Connection/timeout failures are wrapped as
        ``ValueError("api unavailable: …")``. Cloud backends use this as
        an existence check; local backends (llama.cpp, KoboldCpp, …)
        override it for their bespoke loading endpoints.
        """
        try:
            await self._load_model_by_existence(model_repo)
        except httpx.HTTPStatusError as e:
            raise ValueError(str(e)) from e
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ValueError(f"api unavailable: {self._api_url}") from e

        self._model = model_repo
        _logger.info("Model set to %s.", self._model)

    async def _load_model_by_existence(self, model_repo: str) -> None:
        if self._model == model_repo:
            return

        available_models = await self.list_models()

        if model_repo in available_models:
            return

        if not available_models:
            raise ValueError(f"model not found: {model_repo}; no models available")

        if len(available_models) > 5:
            raise ValueError(f"model not found: {model_repo}; available models: {', '.join(available_models[:5])}, +{len(available_models) - 1} more")

        raise ValueError(f"model not found: {model_repo}; available models: {', '.join(available_models)}")

    def unload_model(self) -> None:
        """Free resources associated with the loaded model. No-op by default."""

    def offload_model(self) -> None:
        """Offload model to CPU to free GPU memory. No-op by default."""

    def log_usage(self) -> None:
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

    async def aclose(self) -> None:
        await self._async_session.aclose()
