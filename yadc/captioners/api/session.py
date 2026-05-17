import functools
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from urllib.parse import ParseResult, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from yadc.core import logging

from .utils.cache import HTTPResponseCache
from .utils.response_logger import DebugStreamProxy, ResponseLogger

_logger = logging.get_logger(__name__)


class CaptureContext:
    """Holds per-capture state for response logging.

    Yielded by :meth:`Session.capture_response` so callers can inspect or extend
    the context in future batching scenarios.
    """

    __slots__: tuple[str, ...] = ("image_name",)

    def __init__(self, image_name: str):
        self.image_name = image_name


class Session:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        status_forcelist: tuple[int, ...] = (429, 502, 503, 504),
        session: requests.Session | None = None,
        cache: HTTPResponseCache | None = None,
        response_logger: ResponseLogger | None = None,
    ):
        self.base_url = urlparse(base_url.rstrip("/"))
        self.headers = headers or {}

        self.headers["User-Agent"] = self.user_agent

        self._session = session or requests.Session()
        self._setup_retries(max_retries, backoff_factor, status_forcelist)

        self._pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="Thread-api-")
        self._cache = cache
        self._response_logger = response_logger

    @functools.cached_property
    def user_agent(self):
        import yadc  # dependency loop if used at the top

        return f"yadc/{yadc.__version__} (python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})"

    def _setup_retries(self, max_retries: int, backoff_factor: float, status_forcelist: tuple[int, ...]):
        retry_strategy = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            status=max_retries,
            status_forcelist=status_forcelist,
            backoff_factor=backoff_factor,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _create_url(self, path: str):
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

    @contextmanager
    def capture_response(self, *, image_name: str):
        """Context manager that creates a capture context for response logging.

        Yields a :class:`CaptureContext` that must be passed to :meth:`request` /
        :meth:`post` via the ``capture_ctx`` kwarg. No implicit state is stored
        on the session, so each request is explicitly bound to its context — safe
        for future batching.

        # TODO: consider merging capture_response and request into a single
        # context manager to avoid the two-level nesting.

        Args:
            image_name: Optional image identifier to include in the log filename.
        """
        yield CaptureContext(image_name=image_name)

    @contextmanager
    def request(self, method: str, path: str, *, capture_ctx: CaptureContext | None = None, **kwargs):
        assert self._session

        headers = kwargs.pop("headers", {})
        assert isinstance(headers, dict)

        path = self._create_url(path)

        _logger.debug("HTTP Request: %s %s", method, path)

        headers.update(self.headers)

        stream = kwargs.pop("stream", False)

        should_log = capture_ctx is not None and self._response_logger is not None

        # capture request body before it is consumed
        debug_body = kwargs.get("json") if should_log else None

        request = requests.Request(method, url=path, headers=headers, **kwargs)
        session = self._session

        t = self._pool.submit(
            functools.partial(
                session.send,
                session.prepare_request(
                    request,
                ),
                stream=stream,
            )
        )
        response = t.result()

        _logger.debug("HTTP Response: %s %s: %d", method, path, response.status_code)
        _logger.debug("HTTP Response headers: %s %s: %s", method, path, response.headers)
        if not stream:
            _logger.debug("HTTP Response Body: %s %s: %s", method, path, _size_units(len(response.text)))
            _logger.trace("HTTP Response Body: %s %s: %s", method, path, response.text)
        else:
            _logger.debug("HTTP Response Body: %s %s: (streamed)", method, path)

        # set up streaming accumulator for debug logging
        stream_accumulator: list[bytes | str] | None = [] if (should_log and stream) else None

        try:
            if stream_accumulator is not None:
                with response:
                    yield DebugStreamProxy(response, stream_accumulator)
            else:
                # ensure response is closed
                with response:
                    yield response
        finally:
            if should_log:
                assert self._response_logger is not None
                assert capture_ctx is not None

                body = ""
                if stream and stream_accumulator is not None:
                    body = "\n".join(line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line) for line in stream_accumulator)
                else:
                    try:
                        body = response.text
                    except Exception:
                        body = "<failed to read response body>"

                self._response_logger.log(
                    method=method,
                    url=path,
                    request_headers=headers,
                    request_body=debug_body,
                    response_status=response.status_code,
                    response_headers=response.headers,
                    response_body=body,
                    stream=stream,
                    image_name=capture_ctx.image_name,
                )

    @contextmanager
    def get(self, path: str, cache_ttl: float | None = None, **kwargs):
        if self._cache is None or cache_ttl is None:
            with self.request("GET", path, **kwargs) as response:
                yield response
                return

        cached_response = self._cache.get(path)
        if cached_response is not None:
            yield cached_response
            return

        with self.request("GET", path, **kwargs) as response:
            if response.ok:
                assert isinstance(response, requests.Response)
                self._cache.set(path, response, ttl=cache_ttl)

            yield response

    def post(self, path: str, *, capture_ctx: CaptureContext | None = None, **kwargs):
        return self.request("POST", path, capture_ctx=capture_ctx, **kwargs)


_units = ["B", "KiB", "MiB"]


def _size_units(size: int):
    _size = float(size)

    unit = _units[0]
    for unit in _units:
        if _size >= 1024:
            _size /= 1024
            continue

        break

    return f"{_size:.2f} {unit}"
