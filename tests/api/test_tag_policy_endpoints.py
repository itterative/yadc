"""Tests for the per-dataset tag-policy endpoints (``/api/datasets/<name>/tag/policy``).

The controller wires :class:`TagPolicyService` plus
:meth:`DatasetService.get_dataset` (for the registration check) and
``jsonify`` for the response body. Both services are mocked so the
tests cover the wire shape, the 404 (unregistered dataset), and the
body-validation path. The storage round-trip is exercised at the
service test layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from quart import Quart

from yadc.api.controllers.api_tagging import api_tagging
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.taggers.postprocessing import TagPolicy


@pytest.fixture
def mock_datasets() -> MagicMock:
    datasets = MagicMock()
    # ``get_dataset`` returns a truthy sentinel for registered datasets,
    # ``None`` for the 404 path; tests override per case.
    datasets.get_dataset.return_value = object()
    return datasets


@pytest.fixture
def mock_tag_policy() -> MagicMock:
    """A mock ``TagPolicyService`` whose ``get`` returns an empty policy by default.

    Tests that need a populated policy assign ``get.return_value``
    inline; ``set`` is captured for the PUT tests.
    """
    svc = MagicMock()
    svc.get.return_value = TagPolicy()
    return svc


@pytest.fixture
def app(mock_datasets: MagicMock, mock_tag_policy: MagicMock) -> Quart:
    quart_app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    api_tagging(
        bp,
        MagicMock(),  # tagging
        mock_datasets,
        MagicMock(),  # tag_suggestions
        MagicMock(),  # tag_highlights
        mock_tag_policy,
        MagicMock(),  # configuration
        mock_logging,
    )
    quart_app.register_blueprint(bp)
    return quart_app


@pytest.fixture
def client(app: Quart):
    return app.test_client()


class TestGetDatasetPolicy:
    @pytest.mark.asyncio
    async def test_returns_empty_policy_for_unconfigured_dataset(self, client, mock_datasets: MagicMock, mock_tag_policy: MagicMock):
        mock_tag_policy.get.return_value = TagPolicy()
        resp = await client.get("/api/datasets/ds1/tag/policy")
        assert resp.status_code == 200
        assert await resp.get_json() == {"always_add": [], "banned": []}

    @pytest.mark.asyncio
    async def test_returns_stored_lists(self, client, mock_tag_policy: MagicMock):
        mock_tag_policy.get.return_value = TagPolicy(always_add=["a"], banned=["b"])
        resp = await client.get("/api/datasets/ds1/tag/policy")
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data == {"always_add": ["a"], "banned": ["b"]}

    @pytest.mark.asyncio
    async def test_returns_404_for_unregistered_dataset(self, client, mock_datasets: MagicMock, mock_tag_policy: MagicMock):
        mock_datasets.get_dataset.return_value = None
        resp = await client.get("/api/datasets/ghost/tag/policy")
        assert resp.status_code == 404
        assert not mock_tag_policy.get.called


class TestPutDatasetPolicy:
    @pytest.mark.asyncio
    async def test_persists_full_payload(self, client, mock_tag_policy: MagicMock):
        mock_tag_policy.set.return_value = True
        body = {"always_add": ["masterpiece"], "banned": ["nsfw"]}
        resp = await client.put("/api/datasets/ds1/tag/policy", json=body)
        assert resp.status_code == 200
        returned = await resp.get_json()
        assert returned == body
        mock_tag_policy.set.assert_called_once()
        sent = mock_tag_policy.set.call_args.args
        assert sent[0] == "ds1"
        assert isinstance(sent[1], TagPolicy)
        assert sent[1].always_add == ["masterpiece"]
        assert sent[1].banned == ["nsfw"]

    @pytest.mark.asyncio
    async def test_returns_404_for_unregistered_dataset(self, client, mock_datasets: MagicMock, mock_tag_policy: MagicMock):
        mock_datasets.get_dataset.return_value = None
        resp = await client.put(
            "/api/datasets/ghost/tag/policy",
            json={"always_add": [], "banned": []},
        )
        assert resp.status_code == 404
        assert not mock_tag_policy.set.called

    @pytest.mark.asyncio
    async def test_missing_field_returns_400(self, client, mock_tag_policy: MagicMock):
        # ``banned`` is required (no default); omitted → Pydantic 400.
        resp = await client.put(
            "/api/datasets/ds1/tag/policy",
            json={"always_add": []},
        )
        assert resp.status_code == 400
        assert not mock_tag_policy.set.called

    @pytest.mark.asyncio
    async def test_wrong_type_returns_400(self, client, mock_tag_policy: MagicMock):
        resp = await client.put(
            "/api/datasets/ds1/tag/policy",
            json={"always_add": "not a list", "banned": []},
        )
        assert resp.status_code == 400
        assert not mock_tag_policy.set.called

    @pytest.mark.asyncio
    async def test_empty_lists_are_persisted(self, client, mock_tag_policy: MagicMock):
        """Clearing the lists is a real edit — the server passes the empty state through."""
        mock_tag_policy.set.return_value = True
        resp = await client.put(
            "/api/datasets/ds1/tag/policy",
            json={"always_add": [], "banned": []},
        )
        assert resp.status_code == 200
        sent = mock_tag_policy.set.call_args.args[1]
        assert sent.always_add == []
        assert sent.banned == []
