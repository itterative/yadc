"""Tests for the global tag-highlights endpoints (``/api/tagging/highlights``).

The controller maps :class:`TagHighlightsService` get / set to two routes.
The service is mocked so the test covers the wire shape (response body,
``PUT`` validation behaviour, ``400`` on bad input), and the storage
round-trip is exercised at the service test layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from quart import Quart

from yadc.api.controllers.api_tagging import api_tagging
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.api.services.tag_highlights_service import TagHighlights


@pytest.fixture
def mock_tag_highlights() -> MagicMock:
    hl = MagicMock()
    hl.get.return_value = TagHighlights()
    return hl


@pytest.fixture
def app(mock_tag_highlights: MagicMock) -> Quart:
    quart_app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    api_tagging(
        bp,
        MagicMock(),  # tagging
        MagicMock(),  # datasets
        MagicMock(),  # tag_suggestions
        mock_tag_highlights,
        MagicMock(),  # tag_policy
        MagicMock(),  # configuration
        mock_logging,
    )
    quart_app.register_blueprint(bp)
    return quart_app


@pytest.fixture
def client(app: Quart):
    return app.test_client()


class TestGetHighlights:
    @pytest.mark.asyncio
    async def test_returns_defaults_when_empty(self, client, mock_tag_highlights: MagicMock):
        mock_tag_highlights.get.return_value = TagHighlights()
        resp = await client.get("/api/tagging/highlights")
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data == {
            "starred": [],
            "desired": [],
            "undesired": [],
            "category_overrides": {},
        }

    @pytest.mark.asyncio
    async def test_returns_persisted_payload(self, client, mock_tag_highlights: MagicMock):
        mock_tag_highlights.get.return_value = TagHighlights(
            starred=["masterpiece"],
            desired=["cinematic"],
            undesired=["nsfw"],
            category_overrides={"masterpiece": "general"},
        )
        resp = await client.get("/api/tagging/highlights")
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data == {
            "starred": ["masterpiece"],
            "desired": ["cinematic"],
            "undesired": ["nsfw"],
            "category_overrides": {"masterpiece": "general"},
        }


class TestPutHighlights:
    @pytest.mark.asyncio
    async def test_persists_full_payload_and_returns_it(self, client, mock_tag_highlights: MagicMock):
        body = {
            "starred": ["a", "b"],
            "desired": ["c"],
            "undesired": ["d"],
            "category_overrides": {"a": "general"},
        }
        resp = await client.put(
            "/api/tagging/highlights",
            json=body,
        )
        assert resp.status_code == 200
        returned = await resp.get_json()
        assert returned == body
        # Captured call: full payload, verbatim.
        mock_tag_highlights.set.assert_called_once()
        sent = mock_tag_highlights.set.call_args.args[0]
        assert isinstance(sent, TagHighlights)
        assert sent.starred == ["a", "b"]
        assert sent.category_overrides == {"a": "general"}

    @pytest.mark.asyncio
    async def test_missing_field_returns_400(self, client, mock_tag_highlights: MagicMock):
        # ``starred`` is required; omitting it must fail validation.
        resp = await client.put(
            "/api/tagging/highlights",
            json={"desired": [], "undesired": [], "category_overrides": {}},
        )
        assert resp.status_code == 400
        assert not mock_tag_highlights.set.called

    @pytest.mark.asyncio
    async def test_wrong_type_returns_400(self, client, mock_tag_highlights: MagicMock):
        resp = await client.put(
            "/api/tagging/highlights",
            json={
                "starred": "not a list",
                "desired": [],
                "undesired": [],
                "category_overrides": {},
            },
        )
        assert resp.status_code == 400
        assert not mock_tag_highlights.set.called
