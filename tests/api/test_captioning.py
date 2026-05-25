import json
from unittest.mock import MagicMock

import pytest
from flask import Flask

from yadc.api.controllers.api_captioning import api_captioning
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.api.services.captioning import JobInfo


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
    mock_service.get_status.return_value = JobInfo(
        status="idle", dataset_name="test", job_id="", processed=0, total=0, errors=0, error=None
    )

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
    mock_service.get_status.return_value = JobInfo(
        status="running", dataset_name="foo", job_id="abc123", processed=5, total=10, errors=1, error=None
    )

    resp = test_client.get("/api/datasets/foo/caption")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "running"
    assert data["job_id"] == "abc123"
    assert data["processed"] == 5
    assert data["total"] == 10
    assert data["errors"] == 1
