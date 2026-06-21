"""Tests for ``APICaptioner.list_models()`` proxying to the composed client.

The per-backend ``list_models()`` mechanics live on the LLM clients now
(Phase 1b) and are tested in ``tests/llm/``. This file verifies the
captioner delegates correctly and forwards ``cache_ttl``. The env-level
orchestrator that ties this to env-resolution is in ``yadc.cmd.envs.models``
(tested in ``tests/cmd/envs/test_models.py``).
"""

import pytest

from tests.captioners.api.conftest import MockAsyncSession, make_captioner
from yadc.captioners.api.api_captioner import APITypes


def _register_openai_models(session: MockAsyncSession, *, ids: list[str], owned_by: str = "openai") -> None:
    session.register_uri("GET", "models", json={"data": [{"id": i, "object": "model", "owned_by": owned_by} for i in ids]})


class TestProxyDelegation:
    """``APICaptioner.list_models()`` forwards to ``client.list_models()``."""

    @pytest.mark.parametrize("api_type", [APITypes.OPENAI, APITypes.LLAMACPP, APITypes.OLLAMA, APITypes.VLLM, APITypes.OPENROUTER])
    @pytest.mark.asyncio
    async def test_proxies_across_openai_family(self, api_type: APITypes):
        captioner, session = make_captioner(api_type, api_url="mock://api.test/v1")
        _register_openai_models(session, ids=["model-a", "model-b"])

        assert await captioner.list_models() == ["model-a", "model-b"]

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_models(self):
        captioner, session = make_captioner(APITypes.OPENAI, api_url="mock://api.test/v1")
        _register_openai_models(session, ids=[])

        assert await captioner.list_models() == []

    @pytest.mark.asyncio
    async def test_forwards_cache_ttl(self):
        captioner, session = make_captioner(APITypes.OPENAI, api_url="mock://api.test/v1")
        _register_openai_models(session, ids=["m"])

        # cache_ttl is accepted and forwarded without error.
        assert await captioner.list_models(cache_ttl=123.0) == ["m"]

    @pytest.mark.asyncio
    async def test_gemini_proxies_and_filters_to_generate_content(self):
        captioner, session = make_captioner(APITypes.GEMINI, api_url="mock://generativelanguage.googleapis.com/v1beta", api_token="t")
        session.register_uri(
            "GET",
            "models",
            json={
                "models": [
                    {"name": "models/gemini-2.5-flash", "version": "1", "displayName": "F", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/embedding-001", "version": "1", "displayName": "E", "supportedGenerationMethods": ["embedContent"]},
                ]
            },
        )

        assert await captioner.list_models() == ["gemini-2.5-flash"]
