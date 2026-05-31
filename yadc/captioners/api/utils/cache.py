import hashlib
import pathlib
import time
from typing import Any, cast

import httpx
import requests
import requests.structures

from yadc.core import logging

_logger = logging.get_logger(__name__)


class HTTPResponseCache:
    """File-based HTTP response cache.

    Stores responses keyed by URL path. Supports both ``requests.Response``
    (sync) and ``httpx.Response`` (async) via separate getters.

    Note: ``get`` / ``set`` operate synchronously on the filesystem. A future
    async variant may be added when the whole stack moves to async I/O.
    """

    def __init__(self, cache_dir: str | pathlib.Path):
        self.cache_dir: pathlib.Path = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(mode=0o750, exist_ok=True)

    def _key(self, key: str):
        return hashlib.sha256(key.encode()).hexdigest()

    def _is_valid(self, key: str, entry: "_ResponseCacheEntry") -> bool:
        if entry.key != key:
            _logger.debug("HTTP cache miss (bad key)")
            return False

        if entry.expiry == 0:
            return True

        if entry.expiry < time.time():
            _logger.debug("HTTP cache miss (expired)")
            return False

        return True

    def _read_entry(self, key: str) -> "_ResponseCacheEntry | None":
        cache_file = self.cache_dir / self._key(key)

        if not cache_file.exists():
            _logger.debug("HTTP cache miss")
            return None

        with open(cache_file, "r") as f:
            try:
                entry = _ResponseCacheEntry.deserialize(f.read())
            except ValueError:
                _logger.debug("HTTP cache miss (invalid): %s", exc_info=True)
                return None

            if not self._is_valid(key, entry):
                return None

            _logger.debug("HTTP cache hit")
            return entry

        return None  # pyright: ignore[reportUnreachable]

    def get(self, key: str) -> requests.Response | None:
        """Return a cached response as ``requests.Response``."""
        entry = self._read_entry(key)
        if entry is None:
            return None

        response = requests.Response()
        response.status_code = entry.status_code
        response._content = entry.content
        response.headers = requests.structures.CaseInsensitiveDict([(key, value) for key, value in entry.headers.items()])
        response.encoding = response.apparent_encoding

        return response

    def get_httpx(self, key: str) -> httpx.Response | None:
        """Return a cached response as ``httpx.Response``."""
        entry = self._read_entry(key)
        if entry is None:
            return None

        return httpx.Response(
            status_code=entry.status_code,
            headers=entry.headers,
            content=entry.content,
        )

    def set(self, key: str, response: requests.Response | httpx.Response, ttl: float | None = None):
        """Cache a response. Accepts either ``requests.Response`` or ``httpx.Response``."""
        cache_file = self.cache_dir / self._key(key)

        content: bytes
        if isinstance(response, requests.Response):
            content = response.content
        else:
            content = response.content if response._content is not None else b""

        entry = _ResponseCacheEntry(
            key=key,
            expiry=0 if ttl is None else time.time() + ttl,
            status_code=response.status_code,
            headers=dict(response.headers),
            content=content,
        )

        with open(cache_file, "w") as f:
            f.write(entry.serialize())


class _ResponseCacheEntry:
    def __init__(self, key: str, expiry: float, status_code: int, headers: dict[str, str], content: bytes):
        self.key: str = key
        self.expiry: float = expiry
        self.status_code: int = status_code
        self.headers: dict[str, str] = headers
        self.content: bytes = content

    @classmethod
    def deserialize(cls, contents: str):
        import base64
        import json
        import time

        try:
            response: dict[str, Any] = json.loads(contents)

            assert isinstance(response, dict)

            response_key = response.get("key", None)
            response_expiry = response.get("expiry", None)
            response_status_code = response.get("status_code", None)
            response_headers = response.get("headers", None)
            response_content = response.get("content", None)

            assert isinstance(response_key, str)
            assert isinstance(response_expiry, float)
            assert isinstance(response_status_code, int)
            assert isinstance(response_headers, dict)
            response_headers = cast(dict[str, Any], response_headers)
            assert isinstance(response_content, str)

            response_content_raw = base64.b64decode(response_content)
        except Exception as e:
            raise ValueError("bad response cache entry") from e

        response_headers["Date"] = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())

        try:
            return cls(
                key=response_key,
                expiry=response_expiry,
                status_code=response_status_code,
                headers={str(key): str(value) for key, value in response_headers.items()},
                content=response_content_raw,
            )
        except Exception as e:
            raise ValueError("bad response cache entry") from e

    def serialize(self):
        import base64
        import json

        return json.dumps(
            {
                "key": self.key,
                "expiry": self.expiry,
                "status_code": self.status_code,
                "headers": self.headers,
                "content": base64.b64encode(self.content).decode("utf-8"),
            }
        )
