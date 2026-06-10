"""Tests for the ``/api/datasets/<name>/duplicate`` endpoint.

Covers the request validation, mode handling, and the mapping of
:class:`HardlinkNotSupportedError` to HTTP 409 that the frontend
relies on for the "retry as copy" flow.
"""

from unittest.mock import MagicMock

import pytest
from quart import Quart

from yadc.api.configuration import Configuration
from yadc.api.controllers.api_datasets import api_datasets
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.api.services.dataset_repository import DatasetInfo
from yadc.api.services.datasets import HardlinkNotSupportedError


@pytest.fixture
def configuration() -> Configuration:
    return Configuration(
        db_path=":memory:",
        state_path="/tmp/state",
        config_path="/tmp/config.toml",
        cache_path="/tmp/cache",
    )


@pytest.fixture
def app(configuration):
    """Quart test app with the datasets blueprint registered.

    Stashes each mock dependency on ``app`` so the dependent
    fixtures can expose them to tests.
    """
    quart_app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()
    mock_datasets = MagicMock()
    mock_managed = MagicMock()
    mock_upload = MagicMock()
    mock_config_history = MagicMock()
    mock_captioning = MagicMock()
    mock_captioning.is_captioning.return_value = False  # default: not captioning

    api_datasets(
        configuration,
        bp,
        mock_logging,
        mock_datasets,
        mock_managed,
        mock_upload,
        mock_config_history,
        mock_captioning,
    )

    quart_app.register_blueprint(bp)
    quart_app._mock_datasets = mock_datasets
    quart_app._mock_managed = mock_managed
    quart_app._mock_upload = mock_upload
    quart_app._mock_config_history = mock_config_history
    quart_app._mock_captioning = mock_captioning
    return quart_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def mock_datasets(app):
    return app._mock_datasets


@pytest.fixture
def mock_captioning(app):
    return app._mock_captioning


def _dataset_info(name: str) -> DatasetInfo:
    """A minimal ``DatasetInfo`` for the happy-path return value."""
    return DatasetInfo(
        name=name,
        source="upload",
        config_path=f"/tmp/state/datasets/{name}/config.toml",
        image_count=2,
        has_caption=0,
        has_toml=0,
        last_scanned_t=0,
        first_image_id=1,
    )


class TestDuplicateDataset:
    """POST /api/datasets/<name>/duplicate — duplicates a managed dataset."""

    @pytest.mark.asyncio
    async def test_returns_201_on_hardlink_success(self, client, mock_datasets):
        """Happy path with mode='hardlink' returns 201 and the new dataset info."""
        mock_datasets.duplicate_managed_dataset.return_value = _dataset_info("new_ds")

        resp = await client.post(
            "/api/datasets/src_ds/duplicate",
            json={"new_name": "new_ds", "mode": "hardlink"},
        )

        assert resp.status_code == 201
        body = await resp.get_json()
        assert body["name"] == "new_ds"
        assert body["source"] == "upload"
        mock_datasets.duplicate_managed_dataset.assert_called_once_with("src_ds", "new_ds", mode="hardlink")

    @pytest.mark.asyncio
    async def test_returns_201_on_copy_success(self, client, mock_datasets):
        """Happy path with mode='copy' returns 201."""
        mock_datasets.duplicate_managed_dataset.return_value = _dataset_info("new_ds")

        resp = await client.post(
            "/api/datasets/src_ds/duplicate",
            json={"new_name": "new_ds", "mode": "copy"},
        )

        assert resp.status_code == 201
        mock_datasets.duplicate_managed_dataset.assert_called_once_with("src_ds", "new_ds", mode="copy")

    @pytest.mark.asyncio
    async def test_default_mode_is_hardlink_when_not_specified(self, client, mock_datasets):
        """When the body omits ``mode``, the Pydantic model defaults to
        ``hardlink`` and the service is called with that mode."""
        mock_datasets.duplicate_managed_dataset.return_value = _dataset_info("new_ds")

        resp = await client.post(
            "/api/datasets/src_ds/duplicate",
            json={"new_name": "new_ds"},
        )

        assert resp.status_code == 201
        mock_datasets.duplicate_managed_dataset.assert_called_once_with("src_ds", "new_ds", mode="hardlink")

    @pytest.mark.asyncio
    async def test_returns_409_when_hardlink_not_supported(self, client, mock_datasets):
        """When the service raises ``HardlinkNotSupportedError``, the endpoint
        returns 409 with a clear error message — the frontend uses this to
        show the "retry as copy" warning."""
        mock_datasets.duplicate_managed_dataset.side_effect = HardlinkNotSupportedError(
            "Hardlinks are not supported on the target filesystem (/tmp/state/datasets/new_ds)"
        )

        resp = await client.post(
            "/api/datasets/src_ds/duplicate",
            json={"new_name": "new_ds", "mode": "hardlink"},
        )

        assert resp.status_code == 409
        body = await resp.get_json()
        assert "Hardlinks are not supported" in body["error"]

    @pytest.mark.asyncio
    async def test_returns_409_when_captioning_in_progress(self, client, mock_datasets, mock_captioning):
        """If a captioning job is running for the source dataset, the endpoint
        returns 409 without calling the service."""
        mock_captioning.is_captioning.return_value = True

        resp = await client.post(
            "/api/datasets/src_ds/duplicate",
            json={"new_name": "new_ds", "mode": "hardlink"},
        )

        assert resp.status_code == 409
        body = await resp.get_json()
        assert "captioning" in body["error"].lower()
        mock_datasets.duplicate_managed_dataset.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_400_on_value_error(self, client, mock_datasets):
        """Service ``ValueError`` (e.g. name collision, source not managed,
        empty new_name) returns 400 with the error message."""
        mock_datasets.duplicate_managed_dataset.side_effect = ValueError("Dataset 'new_ds' already exists")

        resp = await client.post(
            "/api/datasets/src_ds/duplicate",
            json={"new_name": "new_ds", "mode": "hardlink"},
        )

        assert resp.status_code == 400
        body = await resp.get_json()
        assert "already exists" in body["error"]

    @pytest.mark.asyncio
    async def test_returns_400_when_source_not_managed(self, client, mock_datasets):
        """Duplicating a non-managed (import / create) dataset is rejected
        with 400 by the service's source-validation step."""
        mock_datasets.duplicate_managed_dataset.side_effect = ValueError("Dataset 'ext_ds' is not a managed dataset")

        resp = await client.post(
            "/api/datasets/ext_ds/duplicate",
            json={"new_name": "new_ds", "mode": "hardlink"},
        )

        assert resp.status_code == 400
        body = await resp.get_json()
        assert "not a managed dataset" in body["error"]

    @pytest.mark.asyncio
    async def test_returns_500_on_unexpected_error(self, client, mock_datasets):
        """Any non-HardlinkNotSupportedError exception returns 500 — the
        service's partial-dir cleanup has already run by this point."""
        mock_datasets.duplicate_managed_dataset.side_effect = OSError("disk full")

        resp = await client.post(
            "/api/datasets/src_ds/duplicate",
            json={"new_name": "new_ds", "mode": "hardlink"},
        )

        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_returns_400_on_invalid_body(self, client, mock_datasets):
        """A malformed body (missing ``new_name``) is rejected by Pydantic
        validation with 400. The service is never called."""
        resp = await client.post(
            "/api/datasets/src_ds/duplicate",
            json={"mode": "hardlink"},  # missing new_name
        )

        assert resp.status_code == 400
        mock_datasets.duplicate_managed_dataset.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_400_on_invalid_mode(self, client, mock_datasets):
        """An unknown ``mode`` value is rejected by Pydantic with 400.
        The service is never called."""
        resp = await client.post(
            "/api/datasets/src_ds/duplicate",
            json={"new_name": "new_ds", "mode": "symlink"},
        )

        assert resp.status_code == 400
        mock_datasets.duplicate_managed_dataset.assert_not_called()
