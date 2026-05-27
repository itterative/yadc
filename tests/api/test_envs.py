import json
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from yadc.api.controllers.api_envs import api_envs
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.cmd.config import AppConfigEnv, AppConfigEnvValue
from yadc.cmd.envs.keystorage_password import PasswordRequiredError


@pytest.fixture
def client():
    app = Flask(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    api_envs(bp, mock_logging)

    app.register_blueprint(bp)
    return app.test_client()


def test_get_key_mode_env_password_not_set(client):
    with patch("yadc.api.controllers.api_envs.cmd_config") as mock_config:
        mock_config.load_config.return_value.key_storage.mode = "password"
        with patch("yadc.api.controllers.api_envs.YADC_PASSWORD", None):
            resp = client.get("/api/envs/key-mode")

    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["mode"] == "password"
    assert data["env_password_set"] is False


def test_get_key_mode_env_password_set(client):
    with patch("yadc.api.controllers.api_envs.cmd_config") as mock_config:
        mock_config.load_config.return_value.key_storage.mode = "keyring"
        with patch("yadc.api.controllers.api_envs.YADC_PASSWORD", "secret"):
            resp = client.get("/api/envs/key-mode")

    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["mode"] == "keyring"
    assert data["env_password_set"] is True


def test_list_models_password_required(client):
    env_data = AppConfigEnv(
        api_url=AppConfigEnvValue(value="http://localhost:11434", method="none"),
        api_token=AppConfigEnvValue(value="encrypted-token", method="password"),
        api_model_name=AppConfigEnvValue(value="gemma3", method="none"),
    )

    with patch("yadc.api.controllers.api_envs.cmd_envs") as mock_cmd_envs:
        mock_cmd_envs.get_env.return_value = env_data
        mock_cmd_envs.EncryptionMethod = mock_cmd_envs.EncryptionMethod
        mock_cmd_envs.decrypt_setting.side_effect = PasswordRequiredError("test")
        mock_cmd_envs.PasswordRequiredError = PasswordRequiredError

        resp = client.post("/api/envs/default/models")

    assert resp.status_code == 403
    data = json.loads(resp.data)
    assert data["error"] == "Password required to decrypt environment settings"
    assert data["code"] == "PASSWORD_REQUIRED"
