"""Tests for ``yadc.cmd.envs.models.list_models`` — the orchestrator that
ties env-resolution (``load_env``) to captioner backend dispatch and
per-backend ``list_models()`` methods.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.captioners.api.conftest import MockAsyncSession
from yadc.captioners.api.constants import DEFAULT_LIST_MODELS_TIMEOUT_SECONDS
from yadc.cmd import envs as cmd_envs
from yadc.cmd.envs import models as cmd_envs_models
from yadc.llm.constants import DEFAULT_MODELS_CACHE_TTL_SECONDS

# Patch targets for the cmd module — single source of truth.
_PATCH_LOAD_ENV = "yadc.cmd.envs.models.load_env"


@pytest.fixture
def patched_load_env():
    """Patch ``load_env`` inside ``yadc.cmd.envs.models``.

    ``load_env`` is a synchronous function, so a plain ``MagicMock`` is
    enough. The ``list_models`` orchestrator is async, but it calls
    ``load_env`` directly (no ``await``).
    """
    with patch(_PATCH_LOAD_ENV) as mock:
        yield mock


def _fake_config(url: str = "", token: str = "", model_name: str = "") -> MagicMock:
    """Build a MagicMock that quacks like the ``UserConfig`` returned by ``load_env``."""
    config = MagicMock()
    config.api.url = url
    config.api.token = token
    config.api.model_name = model_name
    return config


class TestListModelsOrchestrator:
    """``cmd_envs.list_models`` decrypts the env, builds an AsyncSession,
    detects the backend, and dispatches to the right inner captioner."""

    @pytest.mark.asyncio
    async def test_returns_sorted_models_for_openai_backend(self, patched_load_env):
        patched_load_env.return_value = _fake_config(url="http://api.openai.com/v1", token="sk-test")

        session = MockAsyncSession()
        session.register_uri(
            "GET",
            "models",
            json={
                "data": [
                    {"id": "zeta", "object": "model", "owned_by": "openai"},
                    {"id": "alpha", "object": "model", "owned_by": "openai"},
                ]
            },
        )

        result = await cmd_envs.list_models("default", async_session=session)

        assert result == ["alpha", "zeta"]
        patched_load_env.assert_called_once_with("default", password=None)

    @pytest.mark.asyncio
    async def test_raises_when_env_has_no_api_url(self, patched_load_env):
        patched_load_env.return_value = _fake_config()

        with pytest.raises(ValueError, match="environment 'default' has no api_url configured"):
            await cmd_envs.list_models("default", async_session=MockAsyncSession())

    @pytest.mark.asyncio
    async def test_forwards_password_to_load_env(self, patched_load_env):
        patched_load_env.return_value = _fake_config(url="http://api.openai.com/v1", token="decrypted")

        session = MockAsyncSession()
        session.register_uri("GET", "models", json={"data": []})

        await cmd_envs.list_models("default", password="hunter2", async_session=session)

        patched_load_env.assert_called_once_with("default", password="hunter2")

    @pytest.mark.asyncio
    async def test_forwards_cache_ttl_to_inner_captioner(self, patched_load_env):
        patched_load_env.return_value = _fake_config(url="http://api.openai.com/v1", token="t")

        session = MockAsyncSession()
        session.register_uri("GET", "models", json={"data": []})

        # No assertion on the inner call's cache_ttl here — that's tested
        # in the captioner package. This just verifies the kwarg is
        # accepted and the call doesn't error.
        result = await cmd_envs.list_models("default", cache_ttl=42.0, async_session=session)

        assert result == []

    @pytest.mark.asyncio
    async def test_default_cache_ttl_matches_constant(self):
        """Sanity: the default kwarg is the shared captioner constant.

        Guards against accidental drift between the cmd layer's default
        and the captioner package's default.
        """
        from inspect import signature

        sig = signature(cmd_envs_models.list_models)
        assert sig.parameters["cache_ttl"].default == DEFAULT_MODELS_CACHE_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_raises_value_error_on_malformed_upstream_response(self, patched_load_env):
        patched_load_env.return_value = _fake_config(url="http://api.openai.com/v1", token="t")

        session = MockAsyncSession()
        session.register_uri("GET", "models", json={"unexpected": "shape"})

        with pytest.raises(ValueError, match="failed to parse model list response"):
            await cmd_envs.list_models("default", async_session=session)

    @pytest.mark.asyncio
    async def test_propagates_httpx_error_on_http_error(self, patched_load_env):
        import httpx

        patched_load_env.return_value = _fake_config(url="http://api.openai.com/v1", token="t")

        session = MockAsyncSession()
        session.register_uri("GET", "models", text="unauthorized", status_code=401)

        with pytest.raises(httpx.HTTPStatusError):
            await cmd_envs.list_models("default", async_session=session)

    @pytest.mark.asyncio
    async def test_default_timeout_matches_constant(self):
        """Sanity: the default kwarg is the shared captioner constant.

        Guards against accidental drift between the cmd layer's default
        timeout and the captioner package's default.
        """
        from inspect import signature

        sig = signature(cmd_envs_models.list_models)
        assert sig.parameters["timeout"].default == DEFAULT_LIST_MODELS_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_raises_timeout_error_when_inner_hangs(self, patched_load_env, monkeypatch):
        """A hung client surfaces ``asyncio.TimeoutError`` so the controller can
        map it to 504. The timeout applies to the *entire* operation
        (API-type inference inside ``create_client`` + ``list_models``), not
        just the list call itself.
        """
        patched_load_env.return_value = _fake_config(url="http://api.openai.com/v1", token="t")

        async def _hang_forever(**_kwargs):
            # Sleep longer than the timeout so ``asyncio.wait_for`` cancels
            # us. The cancellation also closes the session in the
            # orchestrator's ``finally`` block.
            import asyncio as _asyncio

            await _asyncio.sleep(60)
            return ["unreachable"]

        monkeypatch.setattr(cmd_envs_models, "create_client", _hang_forever)

        with pytest.raises(TimeoutError):
            await cmd_envs.list_models("default", timeout=0.1)

    @pytest.mark.asyncio
    async def test_timeout_none_disables_wait_for(self, patched_load_env, monkeypatch):
        """``timeout=None`` skips ``asyncio.wait_for`` entirely — a slow inner
        call is allowed to take its natural time. Useful for callers (e.g.
        CLI scripts) that don't want a hard cap.
        """
        patched_load_env.return_value = _fake_config(url="http://api.openai.com/v1", token="t")

        completed = []

        async def _quick_list_models(*args, **kwargs):
            completed.append(True)
            return []

        # Fake client — the orchestrator only calls ``list_models`` on it.
        class _FakeClient:
            list_models = _quick_list_models

        async def _create_client(**_kwargs):
            return _FakeClient()

        monkeypatch.setattr(cmd_envs_models, "create_client", _create_client)

        # Should not raise despite the inner being slow (it's instant).
        result = await cmd_envs.list_models("default", timeout=None, async_session=MockAsyncSession())

        assert result == []
        assert completed == [True]
