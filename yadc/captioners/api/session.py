import sys
import requests
import functools

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from urllib.parse import urlparse, ParseResult


class Session:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        status_forcelist: tuple[int, ...] = (429, 502, 503, 504),
    ):
        self.base_url = urlparse(base_url.rstrip('/'))
        self.headers = headers or {}

        self.headers['User-Agent'] = self.user_agent

        self._session = requests.Session()
        self._setup_retries(max_retries, backoff_factor, status_forcelist)

        self._pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix='Thread-api-')

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

        headers.update(self.headers)

        stream = kwargs.pop('stream', False)

        request = requests.Request(method, url=self._create_url(path), headers=headers, **kwargs)
        session = self._session

        t = self._pool.submit(functools.partial(session.send, session.prepare_request(request,), stream=stream))
        response = t.result()

        # ensure response is closed
        with response:
            yield response

    def get(self, path: str, **kwargs):
        return self.request('GET', path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request('POST', path, **kwargs)
