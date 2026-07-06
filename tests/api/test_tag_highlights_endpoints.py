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
from yadc.api.services.tag_highlights_service import TaggedEntry, TagHighlights


def _e(name: str, canonical_form: bool = True) -> TaggedEntry:
    return TaggedEntry(name=name, canonical_form=canonical_form)


@pytest.fixture
def mock_configuration() -> MagicMock:
    cfg = MagicMock()
    cfg.tagger_replace_underscores = False
    return cfg


@pytest.fixture
def mock_tag_highlights() -> MagicMock:
    hl = MagicMock()
    hl.get.return_value = TagHighlights()
    return hl


@pytest.fixture
def app(mock_tag_highlights: MagicMock, mock_configuration: MagicMock) -> Quart:
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
        mock_configuration,
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
    async def test_returns_canonical_entries_verbatim(self, client, mock_tag_highlights: MagicMock):
        """Tier entries and override keys come back in their canonical
        ``{name, canonical_form}`` shape — the backend applies no
        display transform (that's the frontend's render concern).
        Known (``canonical_form: True``) and free-text
        (``canonical_form: False``) identities both round-trip
        verbatim."""
        mock_tag_highlights.get.return_value = TagHighlights(
            starred=[_e("speech_bubble"), _e("custom_tag", canonical_form=False)],
            desired=[],
            undesired=[],
            category_overrides={"speech_bubble": "general"},
        )
        resp = await client.get("/api/tagging/highlights")
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["starred"] == [
            {"name": "speech_bubble", "canonical_form": True},
            {"name": "custom_tag", "canonical_form": False},
        ]
        assert data["category_overrides"] == {"speech_bubble": "general"}


class TestPutHighlights:
    @pytest.mark.asyncio
    async def test_persists_full_payload_and_returns_it(self, client, mock_tag_highlights: MagicMock):
        body = {
            "starred": [{"name": "a"}, {"name": "b"}],
            "desired": [{"name": "c"}],
            "undesired": [{"name": "d"}],
            "category_overrides": {"a": "general"},
        }
        resp = await client.put(
            "/api/tagging/highlights",
            json=body,
        )
        assert resp.status_code == 200
        returned = await resp.get_json()
        # Response echoes back the per-entry shape verbatim
        # (``canonical_form`` defaults to ``True`` when omitted from
        # the request); the backend applies no display transform.
        assert returned == {
            "starred": [
                {"name": "a", "canonical_form": True},
                {"name": "b", "canonical_form": True},
            ],
            "desired": [{"name": "c", "canonical_form": True}],
            "undesired": [{"name": "d", "canonical_form": True}],
            "category_overrides": {"a": "general"},
        }
        # Captured call: full payload stored as ``TaggedEntry`` per entry.
        mock_tag_highlights.set.assert_called_once()
        sent = mock_tag_highlights.set.call_args.args[0]
        assert isinstance(sent, TagHighlights)
        assert sent.starred == [TaggedEntry(name="a"), TaggedEntry(name="b")]
        assert sent.category_overrides == {"a": "general"}

    @pytest.mark.asyncio
    async def test_canonical_form_flag_round_trips_verbatim(
        self, client, mock_tag_highlights: MagicMock
    ):
        """A non-canonical (``canonical_form: False``) entry round-trips
        verbatim — the flag survives and the name is untouched (no
        display transform)."""
        body = {
            "starred": [{"name": "custom_tag", "canonical_form": False}],
            "desired": [],
            "undesired": [],
            "category_overrides": {},
        }
        resp = await client.put("/api/tagging/highlights", json=body)
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data == body

    @pytest.mark.asyncio
    async def test_kaomoji_canonical_form_forced_to_false(
        self, client, mock_tag_highlights: MagicMock
    ):
        """A kaomoji name sent with ``canonical_form: True`` is flipped
        to ``False`` in the response — the Pydantic validator is the
        authoritative source for the kaomoji carve-out."""
        body = {
            "starred": [{"name": "^_^", "canonical_form": True}],
            "desired": [],
            "undesired": [],
            "category_overrides": {},
        }
        resp = await client.put("/api/tagging/highlights", json=body)
        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["starred"] == [{"name": "^_^", "canonical_form": False}]

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
        # A bare string where a list of entries is expected.
        resp = await client.put(
            "/api/tagging/highlights",
            json={
                "starred": [{"name": "a"}],
                "desired": "not a list",
                "undesired": [],
                "category_overrides": {},
            },
        )
        assert resp.status_code == 400
        assert not mock_tag_highlights.set.called
