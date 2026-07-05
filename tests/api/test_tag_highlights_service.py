"""Unit tests for :class:`TagHighlightsService`.

Persists the single global highlights blob under
``settings/tagger.tag_highlights``. A missing row returns the
defaults; a corrupt / wrong-shape row is treated as defaults.
"""

from __future__ import annotations

import pytest

from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.settings_repository import SettingsRepository
from yadc.api.services.tag_highlights_service import TagHighlights, TagHighlightsService


@pytest.fixture
def settings_repo(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
) -> SettingsRepository:
    return SettingsRepository(db_connection_factory, logging_factory)


@pytest.fixture
def service(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
    settings_repo: SettingsRepository,
) -> TagHighlightsService:
    return TagHighlightsService(db_connection_factory, logging_factory, settings_repo)


class TestGet:
    def test_returns_defaults_for_missing_row(self, service: TagHighlightsService):
        assert service.get() == TagHighlights()

    def test_round_trips_stored_value(self, service: TagHighlightsService):
        original = TagHighlights(
            starred=["masterpiece", "1girl"],
            desired=["cinematic"],
            undesired=["nsfw"],
            category_overrides={"masterpiece": "general"},
        )
        service.set(original)
        assert service.get() == original

    def test_unparseable_json_returns_defaults(self, service: TagHighlightsService, settings_repo):
        settings_repo.upsert(TagHighlightsService._KEY, "not-json")
        assert service.get() == TagHighlights()

    def test_non_dict_root_returns_defaults(self, service: TagHighlightsService, settings_repo):
        settings_repo.upsert(TagHighlightsService._KEY, "[1, 2, 3]")
        assert service.get() == TagHighlights()

    def test_non_list_tier_falls_back_to_empty(self, service: TagHighlightsService, settings_repo):
        settings_repo.upsert(TagHighlightsService._KEY, '{"starred": "not-a-list"}')
        assert service.get() == TagHighlights()

    def test_non_dict_overrides_falls_back_to_empty(self, service: TagHighlightsService, settings_repo):
        settings_repo.upsert(TagHighlightsService._KEY, '{"category_overrides": []}')
        assert service.get() == TagHighlights()

    def test_partial_payload_keeps_known_fields(self, service: TagHighlightsService, settings_repo):
        """Unrecognised / missing fields are tolerated — the dataclass defaults fill them in."""
        settings_repo.upsert(TagHighlightsService._KEY, '{"starred": ["x"]}')
        result = service.get()
        assert result.starred == ["x"]
        assert result.desired == []
        assert result.undesired == []
        assert result.category_overrides == {}


class TestSet:
    def test_persists_full_payload(self, service: TagHighlightsService, settings_repo):
        original = TagHighlights(
            starred=["a", "b"],
            desired=["c"],
            undesired=["d"],
            category_overrides={"a": "character", "b": "general"},
        )
        service.set(original)
        # Round-trip via the repo's raw form confirms the blob is a
        # single JSON object (the same shape the GET endpoint returns).
        raw = settings_repo.get(TagHighlightsService._KEY)
        import json

        decoded = json.loads(raw)  # type: ignore[arg-type]
        assert decoded == {
            "starred": ["a", "b"],
            "desired": ["c"],
            "undesired": ["d"],
            "category_overrides": {"a": "character", "b": "general"},
        }

    def test_overwrites_existing_value(self, service: TagHighlightsService):
        service.set(TagHighlights(starred=["old"]))
        service.set(TagHighlights(starred=["new"]))
        assert service.get() == TagHighlights(starred=["new"])
