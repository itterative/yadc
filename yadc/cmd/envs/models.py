"""``list_models`` for environments — decrypt the env's token, probe the
upstream ``/models`` endpoint, and return a sorted list of model IDs.

The env-resolution half lives here. The HTTP/parsing half lives in
``yadc.captioners.api`` — per-backend ``list_models()`` methods on
``OpenAICaptioner`` / ``GeminiCaptioner`` / ``KoboldcppCaptioner`` (and
the OpenAI-compatible backends that inherit from ``OpenAICaptioner``).
"""

from __future__ import annotations

from yadc.captioners.api import APICaptioner
from yadc.captioners.api.async_session import AsyncSession
from yadc.captioners.api.constants import DEFAULT_MODELS_CACHE_TTL_SECONDS
from yadc.captioners.api.utils.cache import HTTPResponseCache
from yadc.core import logging

from .envs import load_env

_logger = logging.get_logger(__name__)


async def list_models(
    env: str = "default",
    *,
    password: str | None = None,
    cache: HTTPResponseCache | None = None,
    cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS,
    async_session: AsyncSession | None = None,
) -> list[str]:
    """Return the list of model IDs available on the env's API.

    The env's token is decrypted by :func:`load_env` (which honors
    ``YADC_PASSWORD`` and the explicit ``password`` argument). The actual
    HTTP call and response parsing are delegated to the per-backend
    ``list_models()`` methods on the captioner classes.

    *cache_ttl* controls how long the upstream ``/models`` response is
    cached. Pass ``None`` to disable caching. Defaults to
    :data:`yadc.captioners.api.constants.DEFAULT_MODELS_CACHE_TTL_SECONDS`.

    *async_session* is exposed for testing — a caller can inject a mock
    session to avoid real HTTP calls. When ``None`` (the default), a
    fresh ``AsyncSession`` is created and closed after the call.

    Raises:
        ValueError: when the env has no ``api_url`` configured, or the
            upstream returns a malformed response.
        httpx.HTTPStatusError: on a non-2xx response from the upstream.
        httpx.ConnectError / httpx.TimeoutException: when the upstream is
            unreachable.
    """
    config = load_env(env, password=password)

    if not config.api.url:
        raise ValueError(f"environment '{env}' has no api_url configured")

    _logger.debug("Fetching model list from '%s' for env '%s' (cache_ttl=%s).", config.api.url, env, cache_ttl)

    if async_session is None:
        headers: dict[str, str] = {}
        if config.api.token:
            headers["Authorization"] = f"Bearer {config.api.token}"

        async_session = AsyncSession(config.api.url, headers=headers, cache=cache)
        own_session = True
    else:
        own_session = False

    try:
        captioner = await APICaptioner.create(
            api_url=config.api.url,
            api_token=config.api.token,
            async_session=async_session,
            _warnings=False,
        )
        models = await captioner.list_models(cache_ttl=cache_ttl)
    finally:
        if own_session:
            await async_session.aclose()

    return sorted(models)
