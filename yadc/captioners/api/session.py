import sys
import requests
import functools

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from urllib.parse import urlparse, ParseResult

from yadc.core import logging

from .utils.cache import HTTPResponseCache

_logger = logging.get_logger(__name__)


class Session:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        status_forcelist: tuple[int, ...] = (429, 502, 503, 504),
        session: requests.Session|None = None,
        cache: HTTPResponseCache|None = None,
    ):
        self.base_url = urlparse(base_url.rstrip('/'))
        self.headers = headers or {}

        self.headers['User-Agent'] = self.user_agent

        self._session = session or requests.Session()
        self._setup_retries(max_retries, backoff_factor, status_forcelist)

        self._pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix='Thread-api-')
        self._cache = cache

    @functools.cached_property
    def user_agent(self):
        import yadc # dependency loop if used at the top
        return f'yadc/{yadc.__version__} (python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})'

    def _setup_retries(
        self,
        max_retries: int,
        backoff_factor: float,
        status_forcelist: tuple[int, ...]
    ):
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
        if not path.startswith('/'):
            path = self.base_url.path + '/' + path

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
    def request(self, method: str, path: str, **kwargs):
        assert self._session

        headers = kwargs.pop('headers', {})
        assert isinstance(headers, dict)

        path = self._create_url(path)

        _logger.debug('HTTP Request: %s %s', method, path)

        headers.update(self.headers)

        stream = kwargs.pop('stream', False)

        request = requests.Request(method, url=path, headers=headers, **kwargs)
        session = self._session

        t = self._pool.submit(functools.partial(session.send, session.prepare_request(request,), stream=stream))
        response = t.result()

        _logger.debug('HTTP Response: %s %s: %d', method, path, response.status_code)
        _logger.debug('HTTP Response headers: %s %s: %s', method, path, response.headers)
        if not stream:
            _logger.debug('HTTP Response Body: %s %s: %s', method, path, _size_units(len(response.text)))
            _logger.trace('HTTP Response Body: %s %s: %s', method, path, response.text)
        else:
            _logger.debug('HTTP Response Body: %s %s: (streamed)', method, path)

        # ensure response is closed
        with response:
            yield response

    @contextmanager
    def get(self, path: str, cache_ttl: float|None = None, **kwargs):
        if self._cache is None or cache_ttl is None:
            with self.request('GET', path, **kwargs) as response:
                yield response
                return

        cached_response = self._cache.get(path)
        if cached_response is not None:
            yield cached_response
            return

        with self.request('GET', path, **kwargs) as response:
            if response.ok:
                self._cache.set(path, response, ttl=cache_ttl)
            
            yield response

    def post(self, path: str, **kwargs):
        return self.request('POST', path, **kwargs)


_units = ['B', 'KiB', 'MiB']
def _size_units(size: int):
    _size = float(size)

    unit = _units[0]
    for unit in _units:
        if _size >= 1024:
            _size /= 1024
            continue

        break

    return f'{_size:.2f} {unit}'
