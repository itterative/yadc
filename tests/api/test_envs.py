"""Tests for the /api/envs/* endpoints."""

from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from quart import Quart

from yadc.api.configuration import Configuration
from yadc.api.controllers.api_envs import api_envs
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.cmd.config import AppConfigEnv, AppConfigEnvValue
from yadc.cmd.envs.keystorage_password import PasswordRequiredError

# Patch targets for the controller module. Single source of truth — change
# these here if the controller's imports move.
_PATCH_CMD_CONFIG = "yadc.api.controllers.api_envs.cmd_config"
_PATCH_CMD_ENVS = "yadc.api.controllers.api_envs.cmd_envs"
_PATCH_SET_KEY_MODE = "yadc.api.controllers.api_envs.cmd_envs.set_key_mode"
_PATCH_YADC_PASSWORD = "yadc.api.controllers.api_envs.YADC_PASSWORD"
_PATCH_LIST_MODELS = "yadc.api.controllers.api_envs.cmd_envs.list_models"


def make_app_config_env(
    *,
    api_url: str = "http://localhost:11434",
    api_token: str | None = "token",
    api_model_name: str | None = "gemma3",
    encryption_method: Literal["none", "keyring", "password"] = "none",
) -> AppConfigEnv:
    """Build an ``AppConfigEnv`` with sensible defaults.

    Defaults model the common "happy path" env: a localhost Ollama URL,
    a plaintext token, a model name, and no encryption. Override any
    field to test a different shape (e.g. ``api_model_name=None`` for
    the "no default" case, or ``encryption_method="password"`` for an
    encrypted token).
    """
    return AppConfigEnv(
        api_url=AppConfigEnvValue(value=api_url, method="none"),
        api_token=AppConfigEnvValue(value=api_token, method=encryption_method),
        api_model_name=AppConfigEnvValue(value=api_model_name, method="none"),
    )


@pytest.fixture
def client(test_configuration: Configuration):
    """Quart test client with the envs blueprint registered."""
    app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    api_envs(bp, test_configuration, mock_logging)

    app.register_blueprint(bp)
    return app.test_client()


@pytest.fixture
def patched_cmd_config():
    """Patch the ``cmd_config`` module in the controller."""
    with patch(_PATCH_CMD_CONFIG) as mock_config:
        yield mock_config


@pytest.fixture
def patched_cmd_envs():
    """Patch the ``cmd_envs`` module in the controller."""
    with patch(_PATCH_CMD_ENVS) as mock_cmd_envs:
        yield mock_cmd_envs


@pytest.fixture
def patched_set_key_mode():
    """Patch ``cmd_envs.set_key_mode`` in the controller."""
    with patch(_PATCH_SET_KEY_MODE) as mock:
        yield mock


@pytest.fixture
def yadc_password():
    """Return a context manager that patches ``YADC_PASSWORD`` in the
    controller to a given value. Used by the GET /key-mode tests, where
    the env password's presence is part of the response."""

    def _patch(value):
        return patch(_PATCH_YADC_PASSWORD, value)

    return _patch


@pytest.fixture
def patched_list_models():
    """Patch ``cmd_envs.list_models`` in the controller."""
    with patch(_PATCH_LIST_MODELS, new_callable=AsyncMock) as mock:
        yield mock


class TestGetKeyMode:
    """GET /api/envs/key-mode — returns the current storage mode and whether
    YADC_PASSWORD is set in the backend environment."""

    @pytest.mark.asyncio
    async def test_returns_password_mode_when_env_password_not_set(self, client, patched_cmd_config, yadc_password):
        patched_cmd_config.load_config.return_value.key_storage.mode = "password"
        with yadc_password(None):
            resp = await client.get("/api/envs/key-mode")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["mode"] == "password"
        assert data["env_password_set"] is False

    @pytest.mark.asyncio
    async def test_returns_keyring_mode_when_env_password_set(self, client, patched_cmd_config, yadc_password):
        patched_cmd_config.load_config.return_value.key_storage.mode = "keyring"
        with yadc_password("secret"):
            resp = await client.get("/api/envs/key-mode")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["mode"] == "keyring"
        assert data["env_password_set"] is True


class TestPutKeyMode:
    """PUT /api/envs/key-mode — switches storage mode or changes the password
    for password mode. Wraps ``cmd_envs.set_key_mode`` and maps domain errors
    to the right HTTP status codes."""

    @pytest.mark.asyncio
    async def test_wrong_old_password_returns_403_password_required(self, client, patched_cmd_config, patched_set_key_mode):
        """Wrong current password returns 403 PASSWORD_REQUIRED, not 500 INTERNAL_ERROR.

        The backend distinguishes "you typed the wrong current password" (user
        error) from "something exploded on the server" (server error). The
        frontend relies on the 403 + PASSWORD_REQUIRED code to show a clear
        "Current password is incorrect" message instead of a generic 500.
        """
        patched_cmd_config.load_config.return_value.key_storage.mode = "password"
        patched_set_key_mode.side_effect = PasswordRequiredError("Password-protected private key requires a password to decrypt.")

        resp = await client.put(
            "/api/envs/key-mode",
            json={"mode": "password", "password": "newpass", "old_password": "wrong"},
        )

        assert resp.status_code == 403
        data = await resp.get_json()
        assert data["code"] == "PASSWORD_REQUIRED"
        assert "incorrect" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_password_change_returns_200(self, client, patched_cmd_config, patched_set_key_mode):
        patched_cmd_config.load_config.return_value.key_storage.mode = "password"

        resp = await client.put(
            "/api/envs/key-mode",
            json={"mode": "password", "password": "newpass", "old_password": "oldpass"},
        )

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["mode"] == "password"
        patched_set_key_mode.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_400_bad_request(self, client, patched_cmd_config):
        """Invalid mode value is rejected before any cmd_envs work happens."""
        resp = await client.put(
            "/api/envs/key-mode",
            json={"mode": "invalid", "password": "x"},
        )

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"


class TestListModels:
    """POST (or GET) /api/envs/<name>/models — returns the env's available model list.

    The endpoint accepts both methods. ``POST`` is preferred by the frontend
    because it can carry a ``{"password": "..."}`` body to decrypt a
    password-mode env's token. ``GET`` is kept for the simple case where
    ``YADC_PASSWORD`` is set in the server env.

    Delegates the actual HTTP call to ``cmd_envs.list_models`` (which uses
    the captioner system). The controller is a thin wrapper that maps
    domain errors to the right HTTP status codes and adds the env's
    configured ``api_model_name`` as the default in the response.
    """

    @pytest.mark.asyncio
    async def test_returns_404_when_env_not_found(self, client, patched_cmd_envs):
        patched_cmd_envs.get_env.return_value = None

        resp = await client.post("/api/envs/missing/models", json={})

        assert resp.status_code == 404
        data = await resp.get_json()
        assert data["code"] == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_returns_400_when_env_has_no_api_url(self, client, patched_cmd_envs, patched_list_models):
        """A misconfigured env (no api_url) is a client-fix problem, not an
        upstream error — distinguish it from connection failures by
        returning 400 BAD_REQUEST. The orchestrator must NOT be called.
        """
        env_data = make_app_config_env(api_url="")

        patched_cmd_envs.get_env.return_value = env_data

        resp = await client.post("/api/envs/default/models", json={})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        assert "API URL" in data["error"]
        patched_list_models.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_403_when_token_is_password_encrypted(self, client, patched_cmd_envs, patched_list_models):
        env_data = make_app_config_env(api_token="encrypted-token", encryption_method="password")

        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.side_effect = PasswordRequiredError("test")

        resp = await client.post("/api/envs/default/models", json={})

        assert resp.status_code == 403
        data = await resp.get_json()
        assert data["error"] == "Password required to decrypt environment settings"
        assert data["code"] == "PASSWORD_REQUIRED"

    @pytest.mark.asyncio
    async def test_returns_502_on_connection_error(self, client, patched_cmd_envs, patched_list_models):
        import httpx

        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.side_effect = httpx.ConnectError("connection refused")

        resp = await client.post("/api/envs/default/models", json={})

        assert resp.status_code == 502
        data = await resp.get_json()
        assert data["code"] == "UPSTREAM_ERROR"

    @pytest.mark.asyncio
    async def test_returns_502_on_upstream_http_error(self, client, patched_cmd_envs, patched_list_models):
        import httpx

        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data

        request = httpx.Request("GET", "http://localhost:11434/models")
        response = httpx.Response(401, request=request)
        patched_list_models.side_effect = httpx.HTTPStatusError("unauthorized", request=request, response=response)

        resp = await client.post("/api/envs/default/models", json={})

        assert resp.status_code == 502
        data = await resp.get_json()
        assert data["code"] == "UPSTREAM_ERROR"
        assert "401" in data["error"]

    @pytest.mark.asyncio
    async def test_returns_502_on_value_error(self, client, patched_cmd_envs, patched_list_models):
        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.side_effect = ValueError("bad response shape")

        resp = await client.post("/api/envs/default/models", json={})

        assert resp.status_code == 502
        data = await resp.get_json()
        assert data["code"] == "UPSTREAM_ERROR"
        assert data["error"] == "bad response shape"

    @pytest.mark.asyncio
    async def test_returns_models_and_default(self, client, patched_cmd_envs, patched_list_models):
        """Happy path — env has api_model_name set, captioner returns sorted models."""
        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = ["a-model", "b-model", "c-model"]

        resp = await client.post("/api/envs/default/models", json={})

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["models"] == ["a-model", "b-model", "c-model"]
        assert data["default"] == "gemma3"

    @pytest.mark.asyncio
    async def test_omits_default_when_no_model_name_configured(self, client, patched_cmd_envs, patched_list_models):
        env_data = make_app_config_env(api_model_name=None)
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = ["m1", "m2"]

        resp = await client.post("/api/envs/default/models", json={})

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["models"] == ["m1", "m2"]
        assert "default" not in data

    @pytest.mark.asyncio
    async def test_passes_cache_ttl_from_configuration(self, client, patched_cmd_envs, patched_list_models, test_configuration: Configuration):
        """The cache_ttl kwarg forwarded to ``cmd_envs.list_models`` matches
        ``Configuration.api_models_cache_ttl``."""
        env_data = make_app_config_env(api_model_name=None)
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = []

        resp = await client.post("/api/envs/default/models", json={})

        assert resp.status_code == 200
        patched_list_models.assert_awaited_once()
        _args, kwargs = patched_list_models.call_args
        # first positional arg is the env name; cache_ttl is a kwarg
        assert kwargs.get("cache_ttl") == test_configuration.api_models_cache_ttl
        # And the configured value should be the default 5-minute TTL.
        assert test_configuration.api_models_cache_ttl == 300.0

    @pytest.mark.asyncio
    async def test_post_forwards_password_to_cmd_envs(self, client, patched_cmd_envs, patched_list_models):
        """A password in the POST body is forwarded to ``cmd_envs.list_models``
        so the env's token can be decrypted when ``YADC_PASSWORD`` is unset."""
        env_data = make_app_config_env(api_token="encrypted-token", encryption_method="password")
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = ["m1"]

        resp = await client.post("/api/envs/default/models", json={"password": "hunter2"})

        assert resp.status_code == 200
        patched_list_models.assert_awaited_once()
        _args, kwargs = patched_list_models.call_args
        assert kwargs.get("password") == "hunter2"

    @pytest.mark.asyncio
    async def test_post_without_password_uses_none(self, client, patched_cmd_envs, patched_list_models):
        """No password in the body means ``password=None`` is forwarded, which
        lets ``cmd_envs.list_models`` fall back to ``YADC_PASSWORD`` (or fail
        with ``PasswordRequiredError`` if neither is set)."""
        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = ["m1"]

        resp = await client.post("/api/envs/default/models", json={})

        assert resp.status_code == 200
        patched_list_models.assert_awaited_once()
        _args, kwargs = patched_list_models.call_args
        assert kwargs.get("password") is None

    @pytest.mark.asyncio
    async def test_get_still_works_without_password(self, client, patched_cmd_envs, patched_list_models):
        """Regression: the GET path still works (no body, no password) so
        callers that don't have a body to send — e.g. browser preflight
        or the ``YADC_PASSWORD``-env-var case — keep functioning."""
        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = ["m1"]

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 200
        patched_list_models.assert_awaited_once()
        _args, kwargs = patched_list_models.call_args
        assert kwargs.get("password") is None
