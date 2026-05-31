from contextlib import asynccontextmanager
from typing import Any, Callable

import httpx
import pytest

from yadc.captioners.api.async_session import AsyncSession


@pytest.fixture
def load_test_data():
    def _load_test_data(case: str):
        import pathlib

        package_root = pathlib.Path(__file__).parent

        with open(package_root / "test_data" / case, "r") as f:
            return f.read()

    return _load_test_data


class MockAsyncSession(AsyncSession):
    """Test helper that yields pre-baked ``httpx.Response`` bodies instead of
    making real HTTP calls.

    Use :meth:`register_uri` to declaratively match routes (similar to
    ``requests_mock``).  Unmatched requests raise ``AssertionError``.
    """

    def __init__(self):
        # Dummy URL — never used because we override arequest.
        super().__init__("http://test")
        self._routes: list[tuple[str, str, str | Callable[..., str] | None, Any | Callable[..., Any] | None, int]] = []

    def register_uri(
        self,
        method: str,
        path: str,
        *,
        text: str | Callable[..., str] | None = None,
        json: Any | Callable[..., Any] | None = None,
        status_code: int = 200,
    ) -> None:
        """Register a response for an exact ``method`` + ``path`` pair.

        *path* should be the relative path passed to the session (e.g.
        ``"models"`` or ``"chat/completions"``).

        *text* or *json* may be callables.  Callables receive
        ``(method, path, **request_kwargs)`` and should return the response
        body.
        """
        self._routes.append((method.upper(), path.lstrip("/"), text, json, status_code))

    @asynccontextmanager
    async def arequest(self, method: str, path: str, *, capture_ctx=None, **kwargs):
        path = path.lstrip("/")
        for route_method, route_path, route_text, route_json, route_status in self._routes:
            if route_method == method.upper() and route_path == path:
                request = httpx.Request(method, f"http://test/{path}")
                body_text = route_text
                body_json = route_json
                if callable(body_text):
                    body_text = body_text(method, path, **kwargs)
                if callable(body_json):
                    body_json = body_json(method, path, **kwargs)
                if body_json is not None:
                    yield httpx.Response(route_status, json=body_json, request=request)
                elif body_text is not None:
                    yield httpx.Response(route_status, text=body_text, request=request)
                else:
                    raise AssertionError(f"Registered route {method} {path} has no text or json")
                return

        raise AssertionError(f"Unexpected request: {method} {path}")

    @asynccontextmanager
    async def post(self, path: str, *, capture_ctx=None, **kwargs):
        async with self.arequest("POST", path, capture_ctx=capture_ctx, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def get(self, path: str, cache_ttl=None, **kwargs):
        async with self.arequest("GET", path, **kwargs) as response:
            yield response

    async def aclose(self) -> None:
        pass


@pytest.fixture
def mock_async_session():
    return MockAsyncSession()
