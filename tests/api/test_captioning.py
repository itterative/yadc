import json
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from yadc.api.controllers.api_captioning import api_captioning
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.api.services.captioning import JobInfo
from yadc.cmd.envs.keystorage_password import PasswordRequiredError


@pytest.fixture
def client():
    app = Flask(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()
    mock_service = MagicMock()

    api_captioning(bp, mock_logging, mock_service)

    app.register_blueprint(bp)
    return app.test_client(), mock_service


def test_get_captioning_status_idle(client):
    test_client, mock_service = client
    mock_service.get_status.return_value = JobInfo(status="idle", dataset_name="test", job_id="", processed=0, total=0, errors=0, error=None)

    resp = test_client.get("/api/datasets/test/caption")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "idle"
    assert data["dataset_name"] == "test"
    assert data["job_id"] == ""
    assert data["processed"] == 0
    assert data["total"] == 0


def test_get_captioning_status_running(client):
    test_client, mock_service = client
    mock_service.get_status.return_value = JobInfo(status="running", dataset_name="foo", job_id="abc123", processed=5, total=10, errors=1, error=None)

    resp = test_client.get("/api/datasets/foo/caption")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "running"
    assert data["job_id"] == "abc123"
    assert data["processed"] == 5
    assert data["total"] == 10
    assert data["errors"] == 1


def test_start_captioning_password_required(client):
    test_client, mock_service = client

    with patch("yadc.api.controllers.api_captioning.cmd_envs") as mock_cmd_envs:
        mock_cmd_envs.PasswordRequiredError = PasswordRequiredError
        mock_cmd_envs.load_env.side_effect = PasswordRequiredError("test")

        resp = test_client.post("/api/datasets/test/caption", data=json.dumps({}), content_type="application/json")

    assert resp.status_code == 403
    data = json.loads(resp.data)
    assert data["error"] == "Password required to decrypt environment settings"
    assert data["code"] == "PASSWORD_REQUIRED"
    mock_service.start_job.assert_not_called()


def test_caption_single_image_password_required(client):
    test_client, mock_service = client

    with patch("yadc.api.controllers.api_captioning.cmd_envs") as mock_cmd_envs:
        mock_cmd_envs.PasswordRequiredError = PasswordRequiredError
        mock_cmd_envs.load_env.side_effect = PasswordRequiredError("test")

        resp = test_client.post("/api/datasets/test/images/1/caption", data=json.dumps({}), content_type="application/json")

    assert resp.status_code == 403
    data = json.loads(resp.data)
    assert data["error"] == "Password required to decrypt environment settings"
    assert data["code"] == "PASSWORD_REQUIRED"
    mock_service.caption_single.assert_not_called()


def test_caption_single_image_returns_job_info(client):
    test_client, mock_service = client
    mock_service.caption_single.return_value = JobInfo(
        status="running", dataset_name="test", job_id="abc123", processed=0, total=1, errors=0, error=None
    )

    with patch("yadc.api.controllers.api_captioning.cmd_envs") as mock_cmd_envs:
        mock_cmd_envs.PasswordRequiredError = PasswordRequiredError
        resp = test_client.post("/api/datasets/test/images/1/caption", data=json.dumps({}), content_type="application/json")

    assert resp.status_code == 202
    data = json.loads(resp.data)
    assert data["status"] == "running"
    assert data["job_id"] == "abc123"
    assert data["total"] == 1
    mock_service.caption_single.assert_called_once()
