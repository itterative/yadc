"""``create_client`` — URL + probe based routing to the right client class."""

import pytest

from yadc.llm import (
    GeminiLLMClient,
    KoboldcppLLMClient,
    LlamacppLLMClient,
    OpenAILLMClient,
    OpenRouterLLMClient,
    VllmLLMClient,
    create_client,
)


class TestUrlRouting:
    """Domain-based routing needs no probing."""

    @pytest.mark.asyncio
    async def test_openai_domain(self, make_session):
        client = await create_client(api_url="https://api.openai.com/v1", api_token="t", async_session=make_session())
        assert isinstance(client, OpenAILLMClient)

    @pytest.mark.asyncio
    async def test_openrouter_domain(self, make_session):
        client = await create_client(api_url="https://openrouter.ai/api/v1", api_token="t", async_session=make_session())
        assert isinstance(client, OpenRouterLLMClient)

    @pytest.mark.asyncio
    async def test_gemini_domain(self, make_session):
        client = await create_client(api_url="https://generativelanguage.googleapis.com/v1beta", api_token="t", async_session=make_session())
        assert isinstance(client, GeminiLLMClient)

    @pytest.mark.asyncio
    async def test_vertex_domain(self, make_session):
        client = await create_client(api_url="https://us-central1-aiplatform.googleapis.com/v1beta1", api_token="t", async_session=make_session())
        assert isinstance(client, GeminiLLMClient)


class TestProbeRouting:
    """``/models`` ``owned_by`` signals identify the local OpenAI-compatible servers."""

    @pytest.mark.asyncio
    async def test_llamacpp_via_owned_by(self, make_session):
        session = make_session()
        session.register_uri("GET", "models", json={"data": [{"id": "x", "owned_by": "llamacpp"}]})

        client = await create_client(api_url="http://localhost:8080/v1", api_token="t", async_session=session)
        assert isinstance(client, LlamacppLLMClient)

    @pytest.mark.asyncio
    async def test_koboldcpp_via_owned_by(self, make_session):
        session = make_session()
        session.register_uri("GET", "models", json={"data": [{"id": "x", "owned_by": "koboldcpp"}]})

        client = await create_client(api_url="http://localhost:5001/v1", api_token="t", async_session=session)
        assert isinstance(client, KoboldcppLLMClient)

    @pytest.mark.asyncio
    async def test_vllm_via_owned_by(self, make_session):
        session = make_session()
        session.register_uri("GET", "models", json={"data": [{"id": "x", "owned_by": "vllm"}]})

        client = await create_client(api_url="http://localhost:8000/v1", api_token="t", async_session=session)
        assert isinstance(client, VllmLLMClient)

    @pytest.mark.asyncio
    async def test_unknown_server_defaults_to_openai(self, make_session):
        # No domain match, no owned_by signal, and all health/version probes miss
        # (the mock raises AssertionError for them, which _infer_api_type catches).
        session = make_session()
        session.register_uri("GET", "models", json={"data": [{"id": "x", "owned_by": "default"}]})

        client = await create_client(api_url="http://localhost:9999/v1", api_token="t", async_session=session)
        assert isinstance(client, OpenAILLMClient)


class TestValidation:
    @pytest.mark.asyncio
    async def test_no_api_url_raises(self, make_session):
        with pytest.raises(ValueError, match="no api_url"):
            await create_client(api_url="", async_session=make_session())
