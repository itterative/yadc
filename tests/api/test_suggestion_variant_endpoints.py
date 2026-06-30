"""Tests for the tag-suggestion variant endpoints.

Routes covered:

- ``GET /api/tagging/suggest/variant`` — return the active variant, the
  config default, and the full variant list in one payload.
- ``PUT /api/tagging/suggest/variant`` — coerce + persist a new variant;
  400 on an unknown value.

The controller dependencies are mocked (same shape as
``tests/api/test_tagger_swap_endpoints.py``). ``set_variant`` is an
``AsyncMock`` so the PUT endpoint can await it; the controller drives
the enum coercion + response shaping itself.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from quart import Quart

from yadc.api.controllers.api_tagging import api_tagging
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.api.services.tag_suggestions import CatalogVariant


@pytest.fixture
def mock_tag_suggestions() -> MagicMock:
    tag_suggestions = MagicMock()
    # Default to the NoobAIXL active variant; individual tests override.
    tag_suggestions.variant = CatalogVariant.NOOBAIXL
    tag_suggestions.set_variant = AsyncMock()
    return tag_suggestions


@pytest.fixture
def mock_configuration() -> MagicMock:
    configuration = MagicMock()
    configuration.tagger_suggestion_variant = "noobaixl"
    return configuration


@pytest.fixture
def app(mock_tag_suggestions: MagicMock, mock_configuration: MagicMock) -> Quart:
    quart_app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    api_tagging(
        bp,
        MagicMock(),  # tagging
        MagicMock(),  # datasets
        mock_tag_suggestions,
        mock_configuration,
        mock_logging,
    )

    quart_app.register_blueprint(bp)
    return quart_app


@pytest.fixture
def client(app: Quart):
    return app.test_client()


@pytest.mark.asyncio
async def test_get_variant_returns_default_and_list(client, mock_tag_suggestions) -> None:
    """GET surfaces the active variant, the config default, and every known
    variant with its display label."""
    mock_tag_suggestions.variant = CatalogVariant.NOOBAIXL
    resp = await client.get("/api/tagging/suggest/variant")
    assert resp.status_code == 200
    body = await resp.get_json()
    assert body["variant"] == "noobaixl"
    assert body["default"] == "noobaixl"
    values = [v["value"] for v in body["variants"]]
    labels = {v["value"]: v["label"] for v in body["variants"]}
    assert values == ["anima", "illustrious", "noobaixl"]
    # Brand spellings, not title-cased enum values.
    assert labels["noobaixl"] == "NoobAIXL"
    assert labels["illustrious"] == "Illustrious"


@pytest.mark.asyncio
async def test_get_variant_reflects_active_selection(client, mock_tag_suggestions) -> None:
    """``variant`` is whatever the service reports as active — a persisted
    user selection surfaces here, distinct from ``default``."""
    mock_tag_suggestions.variant = CatalogVariant.ILLUSTRIOUS
    resp = await client.get("/api/tagging/suggest/variant")
    body = await resp.get_json()
    assert body["variant"] == "illustrious"
    assert body["default"] == "noobaixl"


@pytest.mark.asyncio
async def test_put_variant_persists_and_returns_new(
    client,
    mock_tag_suggestions,
    mock_configuration,
) -> None:
    """PUT coerces the string to the enum, calls ``set_variant``, and returns
    the new active state."""
    resp = await client.put(
        "/api/tagging/suggest/variant",
        json={"variant": "illustrious"},
    )
    assert resp.status_code == 200
    mock_tag_suggestions.set_variant.assert_awaited_once_with(CatalogVariant.ILLUSTRIOUS)
    body = await resp.get_json()
    assert body["variant"] == "illustrious"
    assert body["default"] == "noobaixl"


@pytest.mark.asyncio
async def test_put_variant_coerces_case_and_whitespace(client, mock_tag_suggestions) -> None:
    """The value is trimmed + lowercased before enum lookup so ``" Noobaixl "``
    lands on the right member."""
    resp = await client.put(
        "/api/tagging/suggest/variant",
        json={"variant": " Noobaixl "},
    )
    assert resp.status_code == 200
    mock_tag_suggestions.set_variant.assert_awaited_once_with(CatalogVariant.NOOBAIXL)


@pytest.mark.asyncio
async def test_put_variant_unknown_returns_400(client, mock_tag_suggestions) -> None:
    """An unknown variant is a 400 — the service never sees a raw string."""
    resp = await client.put(
        "/api/tagging/suggest/variant",
        json={"variant": "flux"},
    )
    assert resp.status_code == 400
    mock_tag_suggestions.set_variant.assert_not_awaited()
