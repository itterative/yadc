"""Unit tests for :class:`TagHighlightsService`.

Persists the single global highlights blob under
``settings/tagger.tag_highlights``. A missing row returns the
defaults; a corrupt / wrong-shape row is treated as defaults.
"""

from __future__ import annotations

import json

import pytest

from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.settings_repository import SettingsRepository
from yadc.api.services.tag_highlights_service import TaggedEntry, TagHighlights, TagHighlightsService


def _e(name: str, canonical_form: bool = True) -> TaggedEntry:
    """Shorthand ``TaggedEntry(name, canonical_form)`` so test data stays readable.

    Defaults to ``canonical_form=True`` (catalog / model-output identity)
    to match the Pydantic default; pass ``False`` for free-text or
    kaomoji entries."""
    return TaggedEntry(name=name, canonical_form=canonical_form)


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
            starred=[_e("masterpiece"), _e("1girl")],
            desired=[_e("cinematic")],
            undesired=[_e("nsfw")],
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
        settings_repo.upsert(TagHighlightsService._KEY, '{"starred": [{"name": "x"}]}')
        result = service.get()
        assert result.starred == [TaggedEntry(name="x", canonical_form=True)]
        assert result.desired == []
        assert result.undesired == []
        assert result.category_overrides == {}


class TestSet:
    def test_persists_full_payload(self, service: TagHighlightsService, settings_repo):
        original = TagHighlights(
            starred=[_e("a"), _e("b")],
            desired=[_e("c")],
            undesired=[_e("d")],
            category_overrides={"a": "character", "b": "general"},
        )
        service.set(original)
        # Round-trip via the repo's raw form confirms the blob is a
        # single JSON object (the same shape the GET endpoint returns).
        raw = settings_repo.get(TagHighlightsService._KEY)
        decoded = json.loads(raw)  # type: ignore[arg-type]
        assert decoded == {
            "starred": [{"name": "a", "canonical_form": True}, {"name": "b", "canonical_form": True}],
            "desired": [{"name": "c", "canonical_form": True}],
            "undesired": [{"name": "d", "canonical_form": True}],
            "category_overrides": {"a": "character", "b": "general"},
        }

    def test_canonical_form_flag_round_trips(self, service: TagHighlightsService, settings_repo):
        """The display-form signal (``canonical_form: False`` for free-text
        entries and kaomojis) is persisted alongside the name — the GET
        boundary needs it to decide whether to apply the user's
        ``replace_underscores`` preference."""
        original = TagHighlights(starred=[_e("speech_bubble"), _e("custom_tag", canonical_form=False)])
        service.set(original)
        assert service.get().starred == original.starred

    def test_kaomoji_canonical_form_overridden_to_false(
        self, service: TagHighlightsService, settings_repo
    ):
        """A kaomoji name sent with ``canonical_form: True`` is flipped to
        ``False`` by the Pydantic validator — the backend is the
        authoritative source for the kaomoji carve-out, so a frontend
        mistake can't accidentally let a kaomoji through with the
        transform applied."""
        original = TagHighlights(starred=[_e("^_^", canonical_form=True)])
        service.set(original)
        assert service.get().starred == [TaggedEntry(name="^_^", canonical_form=False)]

    def test_overwrites_existing_value(self, service: TagHighlightsService):
        service.set(TagHighlights(starred=[_e("old")]))
        service.set(TagHighlights(starred=[_e("new")]))
        assert service.get() == TagHighlights(starred=[_e("new")])
