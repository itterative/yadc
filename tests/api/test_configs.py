import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from flask import Flask

from yadc.api.controllers.api_configs import api_configs
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.api.services.datasets import DatasetInfo


@pytest.fixture
def client():
    app = Flask(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()
    mock_service = MagicMock()

    api_configs(bp, mock_logging, mock_service)

    app.register_blueprint(bp)
    return app.test_client(), mock_service


def _make_temp_config(content: str) -> tuple[str, Path]:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
    tmp.write(content)
    tmp.close()
    return tmp.name, Path(tmp.name)


def test_get_config_returns_validation_error_for_invalid_config(client):
    test_client, mock_service = client
    config_path, _ = _make_temp_config("""
[api]
url = "not-a-url"
model_name = ""

[settings]
max_tokens = 50
image_quality = "best"

[reasoning]
thinking_effort = "extreme"
""")
    mock_service.get_dataset.return_value = DatasetInfo(
        name="test", config_path=config_path
    )

    resp = test_client.get("/api/configs/test")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["name"] == "test"
    assert data["parsed"]["api"]["url"] == "not-a-url"
    assert data["validation_error"] is not None
    assert len(data["validation_error"]) > 0

    # Check that model-level errors are present (validators raise ValueError at model level)
    locs = [tuple(e["loc"]) for e in data["validation_error"]]
    assert ("api",) in locs
    assert ("settings",) in locs
    assert ("reasoning",) in locs


def test_get_config_no_validation_error_for_valid_config(client):
    test_client, mock_service = client
    config_path, _ = _make_temp_config("""
[api]
url = "http://localhost:11434"
model_name = "gemma3"

[settings]
max_tokens = 512
image_quality = "auto"
""")
    mock_service.get_dataset.return_value = DatasetInfo(
        name="test", config_path=config_path
    )

    resp = test_client.get("/api/configs/test")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["validation_error"] is None


def test_patch_config_rejects_invalid_max_tokens(client):
    test_client, mock_service = client
    config_path, _ = _make_temp_config("""
[api]
url = "http://localhost:11434"
model_name = "gemma3"
""")
    mock_service.get_dataset.return_value = DatasetInfo(
        name="test", config_path=config_path
    )

    resp = test_client.patch(
        "/api/configs/test",
        data=json.dumps({"settings": {"max_tokens": 50}}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert data["error"] == "Validation failed"
    assert any(
        e["loc"] == ["settings"] for e in data["details"]
    )


def test_patch_config_rejects_invalid_image_quality(client):
    test_client, mock_service = client
    config_path, _ = _make_temp_config("""
[api]
url = "http://localhost:11434"
model_name = "gemma3"
""")
    mock_service.get_dataset.return_value = DatasetInfo(
        name="test", config_path=config_path
    )

    resp = test_client.patch(
        "/api/configs/test",
        data=json.dumps({"settings": {"image_quality": "best"}}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert any(
        e["loc"] == ["settings"] for e in data["details"]
    )


def test_patch_config_rejects_invalid_rounds(client):
    test_client, mock_service = client
    config_path, _ = _make_temp_config("""
[api]
url = "http://localhost:11434"
model_name = "gemma3"
""")
    mock_service.get_dataset.return_value = DatasetInfo(
        name="test", config_path=config_path
    )

    resp = test_client.patch(
        "/api/configs/test",
        data=json.dumps({"rounds": 0}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert any(e["loc"] == [] for e in data["details"])


def test_patch_config_rejects_nested_type_error(client):
    test_client, mock_service = client
    config_path, _ = _make_temp_config("""
[api]
url = "http://localhost:11434"
model_name = "gemma3"
""")
    mock_service.get_dataset.return_value = DatasetInfo(
        name="test", config_path=config_path
    )

    resp = test_client.patch(
        "/api/configs/test",
        data=json.dumps({"settings": {"max_tokens": "foo"}}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "details" in data
    assert any(
        "max_tokens" in e["loc"] or "settings" in e["loc"] for e in data["details"]
    )


def test_patch_config_accepts_valid_patch(client):
    test_client, mock_service = client
    config_path, path_obj = _make_temp_config("""
[api]
url = "http://localhost:11434"
model_name = "gemma3"
""")
    mock_service.get_dataset.return_value = DatasetInfo(
        name="test", config_path=config_path
    )

    resp = test_client.patch(
        "/api/configs/test",
        data=json.dumps({"rounds": 2, "settings": {"max_tokens": 1024}}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["parsed"]["rounds"] == 2
    assert data["parsed"]["settings"]["max_tokens"] == 1024

    # Verify it was written to disk
    written = path_obj.read_text()
    assert "rounds = 2" in written
    assert "max_tokens = 1024" in written
