import abc
import asyncio
from typing import Any

from yadc.core import Captioner, logging
from yadc.core.prediction import PredictionContext

from .constants import DEFAULT_MODELS_CACHE_TTL_SECONDS
from .utils.cache import HTTPResponseCache
from .utils.response_logger import ResponseLogger

_logger = logging.get_logger(__name__)


class BaseAPICaptioner(Captioner, abc.ABC):
    def __init__(self, **kwargs: Any):
        """
        Initializes the BaseAPICaptioner

        Args:
            api_url (str): Base URL for the API endpoint.
            api_token (str): API key for authentication.

            **kwargs: Optional keyword arguments:
                - `prompt_template` (str): The prompt template used for captioning. If none is provided, the default will be used.
                - `cache` (yadc.captioners.utils.cache.HTTPResponseCache, options): Sets the session cache

        Raises:
            ValueError: If `api_url` is not provided.
        """

        Captioner.__init__(self, **kwargs)

        warnings: bool = kwargs.get("_warnings", True)

        self._api_url: str = kwargs.get("api_url", "")
        self._api_token: str = kwargs.get("api_token", "")

        if not self._api_url:
            raise ValueError("no api_url")

        session_headers: dict[str, str] = {}

        if self._api_token:
            session_headers["Authorization"] = f"Bearer {self._api_token}"
        elif warnings:
            _logger.warning("Warning: no api_token is set, requests will fail if api uses authentication")

        cache: HTTPResponseCache | None = kwargs.get("cache", None)
        assert cache is None or isinstance(cache, HTTPResponseCache)

        response_logger: ResponseLogger | None = kwargs.get("response_logger", None)
        assert response_logger is None or isinstance(response_logger, ResponseLogger)

        from .async_session import AsyncSession

        async_session: AsyncSession | None = kwargs.get("async_session", None)
        assert async_session is None or isinstance(async_session, AsyncSession)

        self._async_session: AsyncSession | None = async_session

        # Guards the per-model `_api_usage` dict. Subclasses (OpenAI,
        # Gemini) write to it from inside `predict_stream` and from the
        # non-streaming `_generate_prediction`; multiple concurrent
        # predictions on the same model would otherwise race on dict
        # mutation. Held for the duration of each individual write, so
        # it doesn't serialise predictions — only the few dict
        # mutations at the end of each streamed / non-streamed response.
        self._api_usage_lock: asyncio.Lock = asyncio.Lock()

    @staticmethod
    def _before_predict(kwargs: dict[str, Any]):
        ctx = kwargs.get("prediction_context", None)
        if ctx is not None:
            assert isinstance(ctx, PredictionContext), f"prediction_context must be a PredictionContext, got {type(ctx)}"
            ctx.reasoning = None
            ctx.reasoning_summary = None
            ctx.reasoning_encrypted = None

    @abc.abstractmethod
    def log_usage(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def list_models(self, cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS) -> list[str]:
        """Return the list of model IDs available on this backend.

        Subclasses override this to query their backend-specific endpoint
        (e.g. ``GET /models`` for OpenAI-compatible, paginated ``models``
        for Gemini, ``/api/admin/list_options`` for Koboldcpp).

        *cache_ttl* is forwarded to the per-request cache lookup so callers
        (e.g. the API's ``Configuration.api_models_cache_ttl``) can tune
        the freshness tradeoff per environment. ``None`` disables caching.
        """
        raise NotImplementedError
