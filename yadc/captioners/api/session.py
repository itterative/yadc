import sys
import requests
import functools

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from urllib.parse import urlparse, ParseResult


class Session:
    _pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='Thread-api-')
    _session: requests.Session|None = None

    def __init__(self, base_url: str, headers: dict[str, str]|None = None):
        self.base_url = urlparse(base_url)
        self.headers = headers or {}

        self.headers['User-Agent'] = self._user_agent

    @property
    def _user_agent(self):
        import yadc # dependency loop if used at the top
        return f'yadc/{yadc.__version__} (python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})'

    def _create_url(self, path: str):
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
        return self.request(path, **kwargs)
    
    def post(self, path: str, **kwargs):
        return self.request(path, **kwargs)
