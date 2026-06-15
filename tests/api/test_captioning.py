"""Tests for the /api/datasets/<name>/caption endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from quart import Quart

from yadc.api.controllers.api_captioning import api_captioning
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.api.services.captioning import JobInfo
from yadc.cmd.envs.keystorage_password import PasswordRequiredError

# Patch target for the controller module. Single source of truth — change
# here if the controller's import moves.
_PATCH_CMD_ENVS = "yadc.api.controllers.api_captioning.cmd_envs"


@pytest.fixture
def app():
    """Quart test app with the captioning blueprint registered.

    Stashes the injected ``CaptioningService`` mock on ``app._mock_service``
    so the dependent fixtures can expose it to tests.
    """
    quart_app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()
    mock_service = MagicMock()

    api_captioning(bp, mock_logging, mock_service)

    quart_app.register_blueprint(bp)
    quart_app._mock_service = mock_service
    return quart_app


@pytest.fixture
def client(app):
    """Quart test client."""
    return app.test_client()


@pytest.fixture
def mock_service(app):
    """The mock ``CaptioningService`` injected into the controller."""
    return app._mock_service


@pytest.fixture
def patched_cmd_envs():
    """Patch ``cmd_envs`` in the controller."""
    with patch(_PATCH_CMD_ENVS) as mock_cmd_envs:
        yield mock_cmd_envs


class TestGetCaptioningStatus:
    """GET /api/datasets/<name>/caption — returns the current captioning job status."""

    @pytest.mark.asyncio
    async def test_returns_idle_status(self, client, mock_service):
        mock_service.get_status_async = AsyncMock(
            return_value=JobInfo(
                status="idle",
                dataset_name="test",
                job_id="",
                processed=0,
                total=0,
                errors=0,
                error=None,
            )
        )

        resp = await client.get("/api/datasets/test/caption")
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["status"] == "idle"
        assert data["dataset_name"] == "test"
        assert data["job_id"] == ""
        assert data["processed"] == 0
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_running_status(self, client, mock_service):
        mock_service.get_status_async = AsyncMock(
            return_value=JobInfo(
                status="running",
                dataset_name="foo",
                job_id="abc123",
                processed=5,
                total=10,
                errors=1,
                error=None,
            )
        )

        resp = await client.get("/api/datasets/foo/caption")
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["status"] == "running"
        assert data["job_id"] == "abc123"
        assert data["processed"] == 5
        assert data["total"] == 10
        assert data["errors"] == 1


class TestStartCaptioning:
    """POST /api/datasets/<name>/caption — starts a captioning job for the whole
    dataset. Surfaces PASSWORD_REQUIRED when the env's token is password-encrypted."""

    @pytest.mark.asyncio
    async def test_password_required_returns_403(self, client, mock_service, patched_cmd_envs):
        patched_cmd_envs.PasswordRequiredError = PasswordRequiredError
        patched_cmd_envs.load_env.side_effect = PasswordRequiredError("test")

        resp = await client.post("/api/datasets/test/caption", json={})

        assert resp.status_code == 403
        data = await resp.get_json()
        assert data["error"] == "Password required to decrypt environment settings"
        assert data["code"] == "PASSWORD_REQUIRED"
        mock_service.start_job_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_preflight_failure_returns_409(self, client, mock_service, patched_cmd_envs):
        """Preflight ``ValueError`` surfaces as 4xx, not 202 + status='error'."""
        patched_cmd_envs.PasswordRequiredError = PasswordRequiredError
        mock_service.start_job_async = AsyncMock(side_effect=ValueError("api model_name must be provided"))

        resp = await client.post("/api/datasets/test/caption", json={})

        assert resp.status_code == 409
        data = await resp.get_json()
        assert data["error"] == "api model_name must be provided"
        assert data["code"] == "CONFLICT"


class TestCaptionSingleImage:
    """POST /api/datasets/<name>/images/<id>/caption — starts a single-image
    captioning job. Surfaces PASSWORD_REQUIRED when the env's token is
    password-encrypted."""

    @pytest.mark.asyncio
    async def test_password_required_returns_403(self, client, mock_service, patched_cmd_envs):
        patched_cmd_envs.PasswordRequiredError = PasswordRequiredError
        patched_cmd_envs.load_env.side_effect = PasswordRequiredError("test")

        resp = await client.post("/api/datasets/test/images/1/caption", json={})

        assert resp.status_code == 403
        data = await resp.get_json()
        assert data["error"] == "Password required to decrypt environment settings"
        assert data["code"] == "PASSWORD_REQUIRED"
        mock_service.start_job_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_202_with_job_info(self, client, mock_service, patched_cmd_envs):
        mock_service.start_job_async = AsyncMock(
            return_value=JobInfo(
                status="running",
                dataset_name="test",
                job_id="abc123",
                processed=0,
                total=1,
                errors=0,
                error=None,
            )
        )
        patched_cmd_envs.PasswordRequiredError = PasswordRequiredError

        resp = await client.post("/api/datasets/test/images/1/caption", json={})

        assert resp.status_code == 202
        data = await resp.get_json()
        assert data["status"] == "running"
        assert data["job_id"] == "abc123"
        assert data["total"] == 1
        mock_service.start_job_async.assert_awaited_once()
        call_args = mock_service.start_job_async.call_args
        assert call_args[0][0] == "test"
        assert call_args[0][1].image_ids == [1]

    @pytest.mark.asyncio
    async def test_preflight_failure_returns_409(self, client, mock_service, patched_cmd_envs):
        """Preflight ``ValueError`` surfaces as 4xx, not 202 + status='error'."""
        patched_cmd_envs.PasswordRequiredError = PasswordRequiredError
        mock_service.start_job_async = AsyncMock(side_effect=ValueError("api model_name must be provided"))

        resp = await client.post("/api/datasets/test/images/1/caption", json={})

        assert resp.status_code == 409
        data = await resp.get_json()
        assert data["error"] == "api model_name must be provided"
        assert data["code"] == "CONFLICT"


class TestDeleteRefineResult:
    """DELETE /api/datasets/<name>/images/<id>/refine — evicts the cached
    refine result after the user accepts it. 200 on success, 409 when
    the cached value doesn't match (a newer refine is cached) or no
    entry is cached, 400 on invalid body."""

    @pytest.mark.asyncio
    async def test_returns_200_on_successful_evict(self, client, mock_service):
        mock_service.evict_refine_result = AsyncMock(return_value=True)

        resp = await client.delete(
            "/api/datasets/test/images/42/refine",
            json={"caption": "accepted text", "source": "caption"},
        )

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["status"] == "ok"
        mock_service.evict_refine_result.assert_awaited_once_with("test", 42, "accepted text", source="caption", draft_name="")

    @pytest.mark.asyncio
    async def test_returns_409_when_service_returns_false(self, client, mock_service):
        """The service returns False for both missing entries and mismatches
        (newer refine cached). Both surface as 409 to the client."""
        mock_service.evict_refine_result = AsyncMock(return_value=False)

        resp = await client.delete(
            "/api/datasets/test/images/42/refine",
            json={"caption": "stale text"},
        )

        assert resp.status_code == 409
        data = await resp.get_json()
        assert data["code"] == "CONFLICT"

    @pytest.mark.asyncio
    async def test_returns_400_on_invalid_source(self, client, mock_service):
        mock_service.evict_refine_result = AsyncMock(return_value=True)

        resp = await client.delete(
            "/api/datasets/test/images/42/refine",
            json={"caption": "accepted text", "source": "bogus"},
        )

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        mock_service.evict_refine_result.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_400_on_invalid_body(self, client, mock_service):
        mock_service.evict_refine_result = AsyncMock()
        resp = await client.delete(
            "/api/datasets/test/images/42/refine",
            json={},  # missing required `caption`
        )

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        mock_service.evict_refine_result.assert_not_awaited()
