"""Tests for the per-backend ``list_models()`` methods.

The orchestrator that ties these together with env-resolution lives in
``yadc.cmd.envs.models`` and is tested in
``tests/cmd/envs/test_models.py``.
"""

from typing import Any

import pytest

from tests.captioners.api.conftest import MockAsyncSession
from yadc.captioners.api import APICaptioner
from yadc.captioners.api.api_captioner import APITypes


def _make_captioner(
    api_type: APITypes,
    *,
    api_url: str = "mock://api.test/v1",
    api_token: str = "api_token",
    session: MockAsyncSession | None = None,
):
    if session is None:
        session = MockAsyncSession()
    return APICaptioner(
        api_type=api_type,
        api_url=api_url,
        api_token=api_token,
        async_session=session,
        _warnings=False,
    ), session


class TestOpenAIListModels:
    """OpenAI-compatible backends (``openai``, ``llamacpp``, ``ollama``,
    ``vllm``, ``openrouter``) all inherit ``OpenAICaptioner.list_models``."""

    @pytest.mark.parametrize(
        "api_type",
        [APITypes.OPENAI, APITypes.LLAMACPP, APITypes.OLLAMA, APITypes.VLLM, APITypes.OPENROUTER],
    )
    @pytest.mark.asyncio
    async def test_returns_model_ids_from_data_field(self, api_type: APITypes):
        captioner, session = _make_captioner(api_type)
        session.register_uri(
            "GET",
            "models",
            json={
                "data": [
                    {"id": "model-a", "object": "model", "owned_by": "tester"},
                    {"id": "model-b", "object": "model", "owned_by": "tester"},
                ]
            },
        )

        result = await captioner.list_models()

        assert result == ["model-a", "model-b"]

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_models(self):
        captioner, session = _make_captioner(APITypes.OPENAI)
        session.register_uri("GET", "models", json={"data": []})

        result = await captioner.list_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_raises_value_error_on_malformed_response(self):
        captioner, session = _make_captioner(APITypes.OPENAI)
        session.register_uri("GET", "models", json={"unexpected": "shape"})

        with pytest.raises(ValueError, match="failed to parse model list response"):
            await captioner.list_models()

    @pytest.mark.asyncio
    async def test_propagates_httpx_error_on_http_error(self):
        import httpx

        captioner, session = _make_captioner(APITypes.OPENAI)
        session.register_uri("GET", "models", text="unauthorized", status_code=401)

        with pytest.raises(httpx.HTTPStatusError):
            await captioner.list_models()


class TestGeminiListModels:
    """Gemini's ``/models`` endpoint paginates via ``nextPageToken`` and
    filters to models supporting ``generateContent``."""

    @pytest.mark.asyncio
    async def test_returns_generate_content_models_only(self):
        captioner, session = _make_captioner(APITypes.GEMINI, api_url="mock://generativelanguage.googleapis.com/v1beta")
        session.register_uri(
            "GET",
            "models",
            json={
                "models": [
                    {
                        "name": "models/gemini-2.5-flash",
                        "version": "001",
                        "displayName": "Gemini 2.5 Flash",
                        "supportedGenerationMethods": ["generateContent", "countTokens"],
                    },
                    {
                        "name": "models/embedding-001",
                        "version": "001",
                        "displayName": "Embedding",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                    {
                        "name": "models/gemini-2.5-pro",
                        "version": "001",
                        "displayName": "Gemini 2.5 Pro",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ]
            },
        )

        result = await captioner.list_models()

        assert result == ["gemini-2.5-flash", "gemini-2.5-pro"]

    @pytest.mark.asyncio
    async def test_paginates_via_next_page_token(self):
        def page1_response(method: str, path: str, **kwargs: Any) -> dict:
            return {
                "models": [
                    {
                        "name": "models/gemini-a",
                        "version": "001",
                        "displayName": "A",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ],
                "nextPageToken": "tok-1",
            }

        def page2_response(method: str, path: str, **kwargs: Any) -> dict:
            assert "pageToken=tok-1" in path, f"expected pageToken in path, got: {path}"
            return {
                "models": [
                    {
                        "name": "models/gemini-b",
                        "version": "001",
                        "displayName": "B",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            }

        session = MockAsyncSession()
        session.register_uri("GET", "models", json=page1_response)
        session.register_uri("GET", "models?pageToken=tok-1", json=page2_response)

        captioner, _ = _make_captioner(APITypes.GEMINI, api_url="mock://generativelanguage.googleapis.com/v1beta", session=session)
        result = await captioner.list_models()

        assert result == ["gemini-a", "gemini-b"]


class TestKoboldcppListModels:
    """Koboldcpp exposes a list of available model filenames via
    ``/api/admin/list_options`` (not the OpenAI-style ``/models``)."""

    @pytest.mark.asyncio
    async def test_returns_list_of_strings_skipping_unload_model(self):
        captioner, session = _make_captioner(APITypes.KOBOLDCPP, api_url="mock://localhost:5001")
        session.register_uri(
            "GET",
            "/api/admin/list_options",
            json=["unload_model", "my-model.gguf", "another-model.gguf"],
        )

        result = await captioner.list_models()

        assert result == ["my-model.gguf", "another-model.gguf"]


class TestAPICaptionerListModels:
    """The instance-level ``APICaptioner.list_models()`` delegates to the
    matched inner captioner. The standalone orchestrator that builds a
    fresh captioner is tested in ``tests/cmd/envs/test_models.py``."""

    @pytest.mark.asyncio
    async def test_delegates_to_inner_captioner(self):
        session = MockAsyncSession()
        session.register_uri(
            "GET",
            "models",
            json={
                "data": [
                    {"id": "model-x", "object": "model", "owned_by": "openai"},
                    {"id": "model-y", "object": "model", "owned_by": "openai"},
                ]
            },
        )

        captioner, _ = _make_captioner(APITypes.OPENAI, session=session)
        result = await captioner.list_models()

        assert result == ["model-x", "model-y"]

    @pytest.mark.asyncio
    async def test_forwards_cache_ttl_to_inner(self):
        session = MockAsyncSession()
        session.register_uri(
            "GET",
            "models",
            json={"data": [{"id": "m", "object": "model", "owned_by": "openai"}]},
        )

        captioner, _ = _make_captioner(APITypes.OPENAI, session=session)
        result = await captioner.list_models(cache_ttl=123.0)

        assert result == ["m"]
