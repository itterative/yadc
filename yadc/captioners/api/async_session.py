import asyncio
import functools
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import ParseResult, urlparse

import httpx

from yadc.core import logging

from .utils.cache import HTTPResponseCache
from .utils.response_logger import ResponseLogger

_logger = logging.get_logger(__name__)


class AsyncCaptureContext:
    """Holds per-capture state for async response logging.

    Mirrors :class:`Session.CaptureContext` so ``ResponseLogger`` can stay
    unchanged.
    """

    image_name: str

    def __init__(self, image_name: str):
        self.image_name = image_name


class _LoggedStreamResponse:
    """Wraps an async streaming response to accumulate lines for debug logging.

    Only exposes the methods the captioners call in streaming mode:
    ``raise_for_status()`` and ``aiter_lines()``.
    """

    _response: httpx.Response
    _accumulator: list[str]

    def __init__(self, response: httpx.Response, accumulator: list[str]):
        self._response = response
        self._accumulator = accumulator

    @property
    def status_code(self) -> int:
        return self._response.status_code

    def raise_for_status(self) -> None:
        self._response.raise_for_status()

    async def aiter_lines(self) -> AsyncGenerator[str, None]:
        async for line in self._response.aiter_lines():
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            self._accumulator.append(text)
            yield text


class AsyncSession:
    """Async HTTP session wrapper around ``httpx.AsyncClient``.

    Mirrors the public API of :class:`Session` so captioners can switch
    between sync and async paths with minimal code duplication.

    Retry behaviour is implemented manually because ``httpx`` does not ship
    with the same built-in retry adapter that ``requests``/``urllib3`` provide.
    """

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        status_forcelist: tuple[int, ...] = (429, 502, 503, 504),
        client: httpx.AsyncClient | None = None,
        cache: HTTPResponseCache | None = None,
        response_logger: ResponseLogger | None = None,
        connect_timeout: float = 30.0,
        read_timeout: float | None = None,
        write_timeout: float = 30.0,
        pool_timeout: float = 30.0,
    ):
        self.base_url: ParseResult = urlparse(base_url.rstrip("/"))
        self.headers: dict[str, str] = headers or {}
        self.headers["User-Agent"] = self.user_agent
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            )
        )
        self._max_retries: int = max_retries
        self._backoff_factor: float = backoff_factor
        self._status_forcelist: tuple[int, ...] = status_forcelist
        self._cache: HTTPResponseCache | None = cache
        self._response_logger: ResponseLogger | None = response_logger

    @functools.cached_property
    def user_agent(self) -> str:
        import yadc  # dependency loop if used at the top

        return f"yadc/{yadc.__version__} (python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})"

    def _create_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = self.base_url.path + "/" + path

        path_result = urlparse(path)

        result = ParseResult(
            scheme=self.base_url.scheme,
            netloc=self.base_url.netloc,
            path=path_result.path,
            params=path_result.params or self.base_url.params,
            query=path_result.query or self.base_url.query,
            fragment=path_result.fragment or self.base_url.fragment,
        )

        return result.geturl()

    async def aclose(self) -> None:
        await self._client.aclose()

    @asynccontextmanager
    async def capture_response(self, *, image_name: str):
        """Context manager that creates a capture context for response logging."""
        yield AsyncCaptureContext(image_name=image_name)

    @asynccontextmanager
    async def request(
        self,
        method: str,
        path: str,
        *,
        capture_ctx: AsyncCaptureContext | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[httpx.Response | _LoggedStreamResponse, None]:
        headers: dict[str, Any] = kwargs.pop("headers", {})
        assert isinstance(headers, dict)

        url = self._create_url(path)

        _logger.debug("HTTP Request: %s %s", method, url)

        headers.update(self.headers)

        stream: Any = kwargs.pop("stream", False)

        should_log = capture_ctx is not None and self._response_logger is not None

        # capture request body before it is consumed
        debug_body: Any = kwargs.get("json") if should_log else None

        response: httpx.Response | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(method, url, headers=headers, **kwargs)
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.TimeoutException) as exc:
                if attempt < self._max_retries:
                    wait = self._backoff_factor * (2**attempt)
                    _logger.debug("HTTP retry in %.1fs after %s: %s %s", wait, type(exc).__name__, method, url)
                    await asyncio.sleep(wait)
                    continue
                raise

            if response.status_code not in self._status_forcelist or attempt == self._max_retries:
                break

            # Retryable status code
            wait = self._backoff_factor * (2**attempt)
            _logger.debug("HTTP retry in %.1fs after %d: %s %s", wait, response.status_code, method, url)
            await asyncio.sleep(wait)

        assert response is not None

        _logger.debug("HTTP Response: %s %s: %d", method, url, response.status_code)
        _logger.debug("HTTP Response headers: %s %s: %s", method, url, response.headers)
        if not stream:
            body_preview = await response.aread()
            _logger.debug("HTTP Response Body: %s %s: %d bytes", method, url, len(body_preview))
            _logger.trace("HTTP Response Body: %s %s: %s", method, url, body_preview.decode("utf-8", errors="replace")[:4096])
        else:
            _logger.debug("HTTP Response Body: %s %s: (streamed)", method, url)

        # set up streaming accumulator for debug logging
        stream_accumulator: list[str] | None = [] if (should_log and stream) else None

        try:
            if stream_accumulator is not None:
                yield _LoggedStreamResponse(response, stream_accumulator)
            else:
                yield response
        finally:
            if should_log:
                assert self._response_logger is not None
                assert capture_ctx is not None

                body = ""
                if stream and stream_accumulator is not None:
                    body = "\n".join(stream_accumulator)
                else:
                    try:
                        body = (await response.aread()).decode("utf-8", errors="replace")
                    except Exception:
                        body = "<failed to read response body>"

                self._response_logger.log(
                    method=method,
                    url=url,
                    request_headers=headers,
                    request_body=debug_body,
                    response_status=response.status_code,
                    response_headers=dict(response.headers),
                    response_body=body,
                    stream=stream,
                    image_name=capture_ctx.image_name,
                )
            await response.aclose()

    @asynccontextmanager
    async def get(self, path: str, cache_ttl: float | None = None, **kwargs: Any):
        if self._cache is None or cache_ttl is None:
            async with self.request("GET", path, **kwargs) as response:
                yield response
                return

        cached_response = self._cache.get_httpx(path)
        if cached_response is not None:
            yield cached_response
            return

        async with self.request("GET", path, **kwargs) as response:
            assert isinstance(response, httpx.Response)
            if response.status_code < 400:
                content = await response.aread()
                self._cache.set(path, response, ttl=cache_ttl)
                # Yield a fresh httpx.Response with the already-read content
                yield httpx.Response(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=content,
                )
            else:
                yield response

    @asynccontextmanager
    async def post(self, path: str, *, capture_ctx: AsyncCaptureContext | None = None, **kwargs: Any):
        async with self.request("POST", path, capture_ctx=capture_ctx, **kwargs) as response:
            yield response
