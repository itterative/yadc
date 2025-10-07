from typing import Dict, Optional

import hashlib
import time
import pathlib

import requests
import requests.structures

from yadc.core import logging

_logger = logging.get_logger(__name__)

class HTTPResponseCache:
    def __init__(self, cache_dir: str|pathlib.Path):
        self.cache_dir: pathlib.Path = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(mode=0o750, exist_ok=True)

    def _key(self, key: str):
        return hashlib.sha256(key.encode()).hexdigest()

    def _is_valid(self, key: str, entry: '_ResponseCacheEntry') -> bool:
        if entry.key != key:
            _logger.debug('HTTP cache miss (bad key)')
            return False

        if entry.expiry == 0:
            return True

        if entry.expiry < time.time():
            _logger.debug('HTTP cache miss (expired)')
            return False

        return True


    def get(self, key: str) -> requests.Response | None:
        cache_file = self.cache_dir / self._key(key)

        if not cache_file.exists():
            _logger.debug('HTTP cache miss')
            return None

        with open(cache_file, "r") as f:
            try:
                entry = _ResponseCacheEntry.deserialize(f.read())
            except ValueError:
                _logger.debug('HTTP cache miss (invalid): %s', exc_info=True)
                return None

            if not self._is_valid(key, entry):
                return None

            _logger.debug('HTTP cache hit')

            response = requests.Response()
            response.status_code = 200
            response._content = entry.content
            response.headers = requests.structures.CaseInsensitiveDict([(key, value) for key, value in entry.headers.items()])
            response.encoding = response.apparent_encoding

            return response

        return None

    def set(self, key: str, response: requests.Response, ttl: float|None = None):
        cache_file = self.cache_dir / self._key(key)

        entry = _ResponseCacheEntry(
            key=key,
            expiry=0 if ttl is None else time.time() + ttl,
            status_code=response.status_code,
            headers=dict(response.headers),
            content=response.content,
        )

        with open(cache_file, "w") as f:
            f.write(entry.serialize())

class _ResponseCacheEntry:
    def __init__(
        self,
        key: str,
        expiry: float,
        status_code: int,
        headers: dict[str, str],
        content: bytes
    ):
        self.key: str = key
        self.expiry: float = expiry
        self.status_code: int = status_code
        self.headers: dict[str, str] = headers
        self.content: bytes = content
    
    @classmethod
    def deserialize(cls, contents: str):
        import json
        import time
        import base64

        try:
            response = json.loads(contents)

            assert isinstance(response, dict)
            
            response_key = response.get('key', None)
            response_expiry = response.get('expiry', None)
            response_status_code = response.get('status_code', None)
            response_headers = response.get('headers', None)
            response_content = response.get('content', None)

            assert isinstance(response_key, str)
            assert isinstance(response_expiry, float)
            assert isinstance(response_status_code, int)
            assert isinstance(response_headers, dict)
            assert isinstance(response_content, str)

            response_content_raw = base64.b64decode(response_content)
        except Exception as e:
            raise ValueError('bad response cache entry') from e

        response_headers['Date'] = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())

        try:
            return cls(
                key=response_key,
                expiry=response_expiry,
                status_code=response_status_code,
                headers={ str(key): str(value) for key, value in response_headers.items() },
                content=response_content_raw,
            )
        except Exception as e:
            raise ValueError('bad response cache entry') from e

    def serialize(self):
        import json
        import base64

        return json.dumps({
            'key': self.key,
            'expiry': self.expiry,
            'status_code': self.status_code,
            'headers': self.headers,
            'content': base64.b64encode(self.content).decode('utf-8'),
        })
