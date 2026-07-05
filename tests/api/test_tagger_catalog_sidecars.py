"""Tests for the catalog-driven ``sidecars`` config.

Three layers:

- :class:`TaggerModelSummary` carries the ``sidecars`` list as part
  of its public schema (catalog-time config).
- ``GET /api/tagger/models`` strips ``sidecars`` from the response
  — it's internal server-side download config, not picker UI.
- :class:`TaggingService._catalog_sidecars` returns the catalog
  entry's list for a known ``repo_id`` and ``[]`` for unknown ones.

The ``OnnxTagger`` best-effort download behaviour is covered by
``tests/taggers/test_onnx.py::TestOnnxTaggerSidecars`` (sits next
to the existing ``fake_model_file`` / ``fake_labels_csv`` fixtures).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yadc.api.controllers.api_tagging import api_tagging
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.api.modules.tagger_catalog import (
    KNOWN_TAGGER_MODELS,
    TaggerModelSummary,
)
from yadc.api.services.tagging import TaggingService


class TestTaggerModelSummarySidecars:
    def test_default_is_empty(self):
        """Curated rows that don't need extra files have an empty ``sidecars``."""
        row = TaggerModelSummary(id="x/y", display="X", params="")
        assert row.sidecars == []

    def test_frozen_prevents_attribute_reassignment(self):
        """``sidecars`` is part of the frozen model — reassignment is rejected."""
        row = TaggerModelSummary(id="x/y", display="X", params="", sidecars=["a.onnx_data"])
        with pytest.raises(Exception):
            row.sidecars = ["b.onnx_data"]  # type: ignore[misc]


class TestCatalogSidecarsEndpoint:
    """``GET /api/tagger/models`` must not expose ``sidecars``."""

    @pytest.fixture
    def app(self):
        """Quart app with the tagging blueprint registered; no service mock needed."""
        from quart import Quart

        quart_app = Quart(__name__)
        bp = ApiBlueprint("api", __name__, url_prefix="/api")
        api_tagging(
            bp,
            MagicMock(),  # tagging
            MagicMock(),  # datasets
            MagicMock(),  # tag_suggestions
            MagicMock(),  # tag_highlights
            MagicMock(),  # tag_policy
            MagicMock(),  # configuration
            MagicMock(),  # logging
        )
        quart_app.register_blueprint(bp)
        return quart_app

    @pytest.mark.asyncio
    async def test_models_response_excludes_sidecars(self, app):
        client = app.test_client()
        resp = await client.get("/api/tagger/models")
        assert resp.status_code == 200
        data = await resp.get_json()
        for row in data["models"]:
            assert "sidecars" not in row
        assert "sidecars" not in data["local"]

    @pytest.mark.asyncio
    async def test_dump_excludes_sidecars_field(self):
        """``Field(exclude=True)`` keeps ``sidecars`` out of the default dump,
        so endpoint code doesn't need a per-call ``exclude={...}``."""
        row = TaggerModelSummary(
            id="x/y",
            display="X",
            params="",
            sidecars=["model.onnx_data"],
        )
        dumped = row.model_dump()
        assert "sidecars" not in dumped


class TestCatalogSidecarsHelper:
    def test_returns_sidecars_for_known_repo(self):
        """A known catalog repo with a configured ``sidecars`` list yields it."""
        from yadc.api.services import tagging as svc_module

        # Use a real entry's id; mutate its sidecars for the test and
        # restore afterwards. Pydantic v2 frozen instances reject
        # attribute reassignment, but ``object.__setattr__`` bypasses
        # the validator and is fine for test setup.
        entry = next(iter(KNOWN_TAGGER_MODELS))
        original = list(entry.sidecars)
        try:
            object.__setattr__(entry, "sidecars", ["model.onnx_data"])
            service = MagicMock(spec=TaggingService)
            helper = svc_module.TaggingService._catalog_sidecars.__get__(service)
            assert helper(entry.id) == ["model.onnx_data"]
        finally:
            object.__setattr__(entry, "sidecars", original)

    def test_returns_empty_for_unknown_repo(self):
        from yadc.api.services import tagging as svc_module

        service = MagicMock(spec=TaggingService)
        helper = svc_module.TaggingService._catalog_sidecars.__get__(service)
        assert helper("nonexistent/repo") == []
