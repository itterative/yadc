from contextlib import asynccontextmanager
from typing import Any, Callable

import httpx
import pytest

from yadc.captioners.api import APICaptioner
from yadc.captioners.api.api_captioner import APITypes
from yadc.llm import (
    GeminiLLMClient,
    KoboldcppLLMClient,
    LlamacppLLMClient,
    OllamaLLMClient,
    OpenAILLMClient,
    OpenRouterLLMClient,
    VllmLLMClient,
)
from yadc.llm.async_session import AsyncSession

_CLIENT_CLASSES: dict[APITypes, type] = {
    APITypes.OPENAI: OpenAILLMClient,
    APITypes.OPENROUTER: OpenRouterLLMClient,
    APITypes.GEMINI: GeminiLLMClient,
    APITypes.LLAMACPP: LlamacppLLMClient,
    APITypes.KOBOLDCPP: KoboldcppLLMClient,
    APITypes.VLLM: VllmLLMClient,
    APITypes.OLLAMA: OllamaLLMClient,
}


def make_captioner(
    api_type: APITypes,
    *,
    api_url: str,
    api_token: str = "api_token",
    session: "MockAsyncSession | None" = None,
    **captioner_kwargs: Any,
) -> tuple[APICaptioner, "MockAsyncSession"]:
    """Build a collapsed ``APICaptioner`` over the right client for *api_type*.

    Replaces the old ``APICaptioner(api_type=..., async_session=...)``
    construction. Returns ``(captioner, session)`` so callers can register
    more routes on the same session.
    """
    if session is None:
        session = MockAsyncSession()
    client_cls = _CLIENT_CLASSES[api_type]
    client = client_cls(api_url=api_url, api_token=api_token, async_session=session, warnings=False)
    captioner = APICaptioner(client=client, **captioner_kwargs)
    return captioner, session


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
        self._routes: list[tuple[str, str, str | Callable[..., str] | None, Any | Callable[..., Any] | None, int, BaseException | None]] = []

    def register_uri(
        self,
        method: str,
        path: str,
        *,
        text: str | Callable[..., str] | None = None,
        json: Any | Callable[..., Any] | None = None,
        status_code: int = 200,
        exc: BaseException | None = None,
    ) -> None:
        """Register a response for an exact ``method`` + ``path`` pair.

        *path* should be the relative path passed to the session (e.g.
        ``"models"`` or ``"chat/completions"``).

        *text* or *json* may be callables.  Callables receive
        ``(method, path, **request_kwargs)`` and should return the response
        body.
        """
        self._routes.append((method.upper(), path.lstrip("/"), text, json, status_code, exc))

    def _match(self, method: str, path: str, **kwargs: Any):
        """Find the registered route for ``method`` + ``path`` and build an ``httpx.Response``.

        *path* should be the relative path passed to the session (e.g.
        ``"models"`` or ``"chat/completions"``).
        """
        path = path.lstrip("/")
        for route_method, route_path, route_text, route_json, route_status, route_exc in self._routes:
            if route_method == method.upper() and route_path == path:
                if route_exc is not None:
                    raise route_exc
                request = httpx.Request(method, f"http://test/{path}")
                body_text = route_text
                body_json = route_json
                if callable(body_text):
                    body_text = body_text(method, path, **kwargs)
                if callable(body_json):
                    body_json = body_json(method, path, **kwargs)
                if body_json is not None:
                    return httpx.Response(route_status, json=body_json, request=request)
                if body_text is not None:
                    return httpx.Response(route_status, text=body_text, request=request)
                raise AssertionError(f"Registered route {method} {path} has no text or json")

        raise AssertionError(f"Unexpected request: {method} {path}")

    @asynccontextmanager
    async def arequest(self, method: str, path: str, *, capture_ctx=None, **kwargs):
        yield self._match(method, path, **kwargs)

    @asynccontextmanager
    async def request(self, method: str, path: str, *, capture_ctx=None, **kwargs):
        # The real ``AsyncSession.request`` is used by API-type inference probes
        # (e.g. ``HEAD /``). Mirror ``arequest`` so unmatched probes raise
        # ``AssertionError`` (which the inference code catches) instead of a
        # real httpx call to the dummy base URL.
        yield self._match(method, path, **kwargs)

    async def open_stream(self, method: str, path: str, *, capture_ctx=None, **kwargs):
        from yadc.llm.async_session import AsyncStreamHandle

        response = self._match(method, path, **kwargs)
        return AsyncStreamHandle(response)

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
