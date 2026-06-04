"""Tests for the /api/envs/* endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from quart import Quart

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


@pytest.fixture
def client():
    """Quart test client with the envs blueprint registered."""
    app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    api_envs(bp, mock_logging)

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
    """POST /api/envs/<name>/models — proxies /models to the env's API.
    Surfaces PASSWORD_REQUIRED when the token is password-encrypted and no
    password is supplied."""

    @pytest.mark.asyncio
    async def test_returns_403_when_token_is_password_encrypted(self, client, patched_cmd_envs):
        env_data = AppConfigEnv(
            api_url=AppConfigEnvValue(value="http://localhost:11434", method="none"),
            api_token=AppConfigEnvValue(value="encrypted-token", method="password"),
            api_model_name=AppConfigEnvValue(value="gemma3", method="none"),
        )

        patched_cmd_envs.get_env.return_value = env_data
        patched_cmd_envs.EncryptionMethod = patched_cmd_envs.EncryptionMethod
        patched_cmd_envs.decrypt_setting.side_effect = PasswordRequiredError("test")
        patched_cmd_envs.PasswordRequiredError = PasswordRequiredError

        resp = await client.post("/api/envs/default/models")

        assert resp.status_code == 403
        data = await resp.get_json()
        assert data["error"] == "Password required to decrypt environment settings"
        assert data["code"] == "PASSWORD_REQUIRED"
