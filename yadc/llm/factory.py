"""``create_client`` — factory that picks the right LLM client for an API endpoint.

Takes the API URL + token (plus optional cache / response logger / session
overrides), infers the backend subtype from the URL (and probes for the
OpenAI-compatible local servers when ambiguous), and returns the matching
:class:`BaseLLMClient` subclass instance. The URL/probe inference is needed
because a single ``api_url`` can point at OpenAI, OpenRouter, Gemini,
llama.cpp, KoboldCpp, vLLM, or Ollama — each with subtly different request
shapes.

Takes primitives (not a config object) so it works for both the env
``UserConfig`` path (``cmd_envs``) and the dataset ``Config`` path (the
captioning runner) without coupling to either config model.

The factory is ``async`` because subtype inference probes the server.
"""

import json
from typing import Any
from urllib.parse import urlparse

import httpx
import pydantic

from .async_session import AsyncSession
from .base import BaseLLMClient
from .cache import HTTPResponseCache
from .gemini import GeminiLLMClient
from .koboldcpp import KoboldcppLLMClient
from .llamacpp import LlamacppLLMClient
from .ollama import OllamaLLMClient
from .openai import OpenAILLMClient
from .openrouter import OpenRouterLLMClient
from .response_logger import ResponseLogger
from .response_models import OpenAIModelsResponse
from .vllm import VllmLLMClient


async def create_client(
    *,
    api_url: str,
    api_token: str = "",
    cache: HTTPResponseCache | None = None,
    response_logger: ResponseLogger | None = None,
    async_session: AsyncSession | None = None,
    **overrides: Any,
) -> BaseLLMClient:
    """Build the right :class:`BaseLLMClient` for ``api_url``.

    Infers the backend subtype (probing the server when the URL alone is
    ambiguous), constructs the matching client, and wires the shared
    session/cache/logger. Pass ``async_session=`` to inject a mock for tests
    (otherwise a fresh :class:`AsyncSession` is created and owned by the
    returned client — close it via ``await client.aclose()``).

    *overrides* are forwarded to the client constructor (timeouts, etc.).
    """
    if not api_url:
        raise ValueError("no api_url")

    if async_session is None:
        headers: dict[str, str] = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        async_session = AsyncSession(api_url, headers=headers, cache=cache, response_logger=response_logger)

    client_cls = await _infer_api_type(async_session, api_url)

    return client_cls(
        api_url=api_url,
        api_token=api_token,
        cache=cache,
        response_logger=response_logger,
        async_session=async_session,
        **overrides,
    )


_OPENAI_DOMAIN = "api.openai.com"
_OPENROUTER_DOMAIN = "openrouter.ai"
_GEMINI_DOMAIN = "generativelanguage.googleapis.com"
_VERTEX_DOMAIN = "-aiplatform.googleapis.com"


async def _infer_api_type(session: AsyncSession, api_url: str) -> type[BaseLLMClient]:
    """Infer the backend subtype by probing the remote server.

    Domain heuristics first (OpenAI / OpenRouter / Gemini / Vertex AI),
    then ``/models`` ``owned_by`` signals, then ``/health`` / serviceinfo /
    version probes for the local servers; falls back to
    :class:`OpenAILLMClient` when nothing matches.
    """
    from typing import cast

    from .response_models import KoboldServiceInfoResponse

    try:
        parsed = urlparse(api_url)

        if parsed.netloc == _OPENAI_DOMAIN:
            return OpenAILLMClient
        if parsed.netloc == _OPENROUTER_DOMAIN:
            return OpenRouterLLMClient
        if parsed.netloc == _GEMINI_DOMAIN:
            return GeminiLLMClient
        if parsed.netloc.endswith(_VERTEX_DOMAIN):
            return GeminiLLMClient
    except Exception:
        pass

    try:
        async with session.get("models") as models_resp:
            models_resp = cast(httpx.Response, models_resp)
            assert models_resp.status_code < 400, f"request failed with http {models_resp.status_code}"

            models_json = models_resp.json()
            assert isinstance(models_json, dict), f"bad models response type; expected dict, got {type(models_json)}"

            models = OpenAIModelsResponse.model_validate(models_json)

            for model in models.data:
                if model.owned_by == "llamacpp":
                    return LlamacppLLMClient
                if model.owned_by == "koboldcpp":
                    return KoboldcppLLMClient
                if model.owned_by == "vllm":
                    return VllmLLMClient
    except AssertionError as e:
        raise ValueError(f"failed to retrieve model list: {e}") from e

    try:
        async with session.get("/health") as health_resp:
            health_resp = cast(httpx.Response, health_resp)
            assert health_resp.status_code < 400
            if "llama.cpp" in health_resp.headers.get("server", "unknown").lower():
                return LlamacppLLMClient
    except Exception:
        pass

    try:
        async with session.get("/.well-known/serviceinfo") as resp:
            resp = cast(httpx.Response, resp)
            assert resp.status_code < 400
            service_info = KoboldServiceInfoResponse.model_validate(resp.json())
            assert service_info.software.name.lower() == "koboldcpp"
            return KoboldcppLLMClient
    except (AssertionError, json.JSONDecodeError, pydantic.ValidationError):
        pass

    try:
        async with session.request("HEAD", "/") as resp:
            resp = cast(httpx.Response, resp)
            assert resp.status_code < 400
            if "ollama" in resp.text.lower():
                return OllamaLLMClient
    except AssertionError:
        pass

    try:
        async with session.get("/api/version") as resp:
            resp = cast(httpx.Response, resp)
            assert resp.status_code < 400
            ollama_json = resp.json()
            assert isinstance(ollama_json, dict) and "version" in ollama_json
        return OllamaLLMClient
    except (AssertionError, json.JSONDecodeError):
        pass

    try:
        async with session.get("/ping") as resp:
            resp = cast(httpx.Response, resp)
            assert resp.status_code < 400
        async with session.post("/ping") as resp:
            resp = cast(httpx.Response, resp)
            assert resp.status_code < 400
        async with session.get("/version") as resp:
            resp = cast(httpx.Response, resp)
            assert resp.status_code < 400
            vllm_json = resp.json()
            assert isinstance(vllm_json, dict) and "version" in vllm_json
        return VllmLLMClient
    except (AssertionError, json.JSONDecodeError):
        pass

    return OpenAILLMClient
