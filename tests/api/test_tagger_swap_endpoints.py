"""Tests for the tagger-model-swap endpoints (``/api/tagger/...``).

Routes covered:

- ``GET /api/tagger/active`` — return the persisted selection + subprocess liveness.
- ``POST /api/tagger/swap`` — drive a swap end-to-end; map
  :class:`TaggerBusyError` to 409 and :class:`TaggerSwapInProgressError`
  to 429 (with ``retry_after_s`` + ``Retry-After`` header).
- ``GET /api/tagger/models`` — return the curated catalog.

The controller dependencies (``TaggingService``, ``DatasetService``,
``Configuration``, ``LoggingFactory``) are wired as ``MagicMock`` —
this is the same shape as ``tests/api/test_captioning.py``. The mock
service has its async methods re-bound to ``AsyncMock`` so the swap
endpoint can drive its real return-value mapping.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from quart import Quart

from yadc.api.controllers.api_tagging import api_tagging
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.api.modules.tagger_catalog import (
    KNOWN_TAGGER_MODELS,
    LOCAL_FILE_ENTRY,
    ActiveTagger,
)
from yadc.api.services.tagging import TaggerBusyError, TaggerSwapInProgressError


@pytest.fixture
def mock_tagging() -> MagicMock:
    """A fully-mocked ``TaggingService``.

    Async methods are re-bound to ``AsyncMock`` so the controller can
    await them; sync properties (``effective_active_tagger``,
    ``is_available``, ``is_configured``) are bare attributes — the
    controller reads them directly via ``tagging.<name>`` (no ``await``).
    """
    tagging = MagicMock()
    tagging.effective_active_tagger = None
    tagging.is_available = False
    tagging.is_configured = False
    tagging.swap_active_model = AsyncMock()
    return tagging


@pytest.fixture
def app(mock_tagging: MagicMock) -> Quart:
    """Quart test app with the tagging blueprint registered."""
    quart_app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()
    mock_datasets = MagicMock()

    api_tagging(bp, mock_tagging, mock_datasets, MagicMock(), MagicMock(), mock_logging)

    quart_app.register_blueprint(bp)
    return quart_app


@pytest.fixture
def client(app: Quart):
    """Quart test client."""
    return app.test_client()


class TestGetActiveTagger:
    """``GET /api/tagger/active`` — surface current selection + subprocess liveness."""

    @pytest.mark.asyncio
    async def test_returns_null_when_nothing_configured(self, client, mock_tagging: MagicMock):
        mock_tagging.effective_active_tagger = None

        resp = await client.get("/api/tagger/active")
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data == {"active": None, "is_available": False}

    @pytest.mark.asyncio
    async def test_returns_synthesized_selection_from_legacy_configuration(self, client, mock_tagging: MagicMock):
        """A legacy Configuration-only setup (no persisted swap but a flat
        ``tagger_model_path``) synthesizes an ``ActiveTagger`` via the
        service's ``effective_active_tagger`` property so the picker sees
        the right model without requiring a no-op swap."""
        synth = ActiveTagger(
            kind="local",
            model_path="/models/animetimm/model.onnx",
            label_path="/models/animetimm/selected_tags.csv",
            preproc_profile="timm",
        )
        mock_tagging.effective_active_tagger = synth
        mock_tagging.is_available = False

        resp = await client.get("/api/tagger/active")
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["active"]["kind"] == "local"
        assert data["active"]["model_path"] == "/models/animetimm/model.onnx"
        assert data["active"]["source"] == "local:/models/animetimm/model.onnx"
        assert data["is_available"] is False

    @pytest.mark.asyncio
    async def test_returns_persisted_selection_with_source_label(self, client, mock_tagging: MagicMock):
        mock_tagging.effective_active_tagger = ActiveTagger(
            kind="hf",
            repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
        )
        mock_tagging.is_available = True

        resp = await client.get("/api/tagger/active")
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["active"]["kind"] == "hf"
        assert data["active"]["repo_id"] == "SmilingWolf/wd-eva02-large-tagger-v3"
        assert data["active"]["source"] == "hf:SmilingWolf/wd-eva02-large-tagger-v3"
        assert data["is_available"] is True

    @pytest.mark.asyncio
    async def test_reports_is_available_false_when_subprocess_down(self, client, mock_tagging: MagicMock):
        mock_tagging.effective_active_tagger = ActiveTagger(kind="hf", repo_id="SmilingWolf/wd-vit-tagger-v3")
        mock_tagging.is_available = False

        resp = await client.get("/api/tagger/active")
        data = await resp.get_json()
        assert data["active"] is not None
        assert data["is_available"] is False


class TestSwapTagger:
    """``POST /api/tagger/swap`` — drive a swap; map service exceptions to HTTP."""

    @pytest.mark.asyncio
    async def test_successful_swap_returns_new_active_payload(self, client, mock_tagging: MagicMock):
        selection = ActiveTagger(kind="hf", repo_id="SmilingWolf/wd-eva02-large-tagger-v3")
        mock_tagging.swap_active_model = AsyncMock(return_value=selection)
        mock_tagging.effective_active_tagger = selection
        mock_tagging.is_available = True

        resp = await client.post(
            "/api/tagger/swap",
            json=selection.model_dump(),
        )
        assert resp.status_code == 202
        data = await resp.get_json()
        assert data["active"]["repo_id"] == "SmilingWolf/wd-eva02-large-tagger-v3"
        assert data["active"]["source"] == "hf:SmilingWolf/wd-eva02-large-tagger-v3"
        assert data["is_available"] is True
        mock_tagging.swap_active_model.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_busy_error_maps_to_409(self, client, mock_tagging: MagicMock):
        mock_tagging.swap_active_model = AsyncMock(side_effect=TaggerBusyError("a batch tagging job is running on ['ds']"))

        resp = await client.post(
            "/api/tagger/swap",
            json=ActiveTagger(kind="hf", repo_id="SmilingWolf/wd-vit-tagger-v3").model_dump(),
        )
        assert resp.status_code == 409
        data = await resp.get_json()
        assert "batch tagging job is running" in data["error"]
        assert data["code"] == "CONFLICT"

    @pytest.mark.asyncio
    async def test_swap_in_progress_maps_to_429_with_retry_hint(self, client, mock_tagging: MagicMock):
        """The 429 body carries ``retry_after_s`` and the standard ``Retry-After``
        header so the picker UI can show a "retry in 2s" hint."""
        mock_tagging.swap_active_model = AsyncMock(side_effect=TaggerSwapInProgressError("another swap is already in progress", retry_after_s=2.0))

        resp = await client.post(
            "/api/tagger/swap",
            json=ActiveTagger(kind="hf", repo_id="SmilingWolf/wd-vit-tagger-v3").model_dump(),
        )
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "2"

        data = json.loads(await resp.get_data(as_text=True))
        assert "another swap" in data["error"]
        assert data["retry_after_s"] == 2.0
        assert data["code"] == "TOO_MANY_REQUESTS"

    @pytest.mark.asyncio
    async def test_invalid_body_returns_400(self, client, mock_tagging: MagicMock):
        """``kind='hf'`` without a ``repo_id`` fails the model's validator — we
        surface that as a 400 (not letting the swap proceed with garbage)."""
        resp = await client.post("/api/tagger/swap", json={"kind": "hf"})
        assert resp.status_code == 400
        mock_tagging.swap_active_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_kind_returns_400(self, client, mock_tagging: MagicMock):
        resp = await client.post("/api/tagger/swap", json={"kind": "weird", "repo_id": "x/y"})
        assert resp.status_code == 400
        mock_tagging.swap_active_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_local_kind_with_empty_path_returns_400(self, client, mock_tagging: MagicMock):
        """The validator's ``kind='local'`` + empty ``model_path`` rejection is
        treated as a 400 — same path as the HF empty-repo case."""
        resp = await client.post("/api/tagger/swap", json={"kind": "local", "model_path": ""})
        assert resp.status_code == 400
        mock_tagging.swap_active_model.assert_not_awaited()


class TestListTaggerModels:
    """``GET /api/tagger/models`` — static curated catalog."""

    @pytest.mark.asyncio
    async def test_returns_curated_models_and_local_sentinel(self, client):
        resp = await client.get("/api/tagger/models")
        assert resp.status_code == 200
        data = await resp.get_json()
        assert [m["id"] for m in data["models"]] == [m.id for m in KNOWN_TAGGER_MODELS]
        assert data["models"][0]["display"] == KNOWN_TAGGER_MODELS[0].display
        assert data["local"] == LOCAL_FILE_ENTRY.model_dump()
        assert data["local"]["id"] == "__local__"
        # ``profiles`` mirrors :func:`list_profiles` so the picker's
        # dropdown stays in sync without a frontend constant.
        from yadc.taggers.onnx_preprocess import list_profiles

        assert data["profiles"] == list_profiles()
