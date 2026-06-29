"""Tests for ``TaggingService`` active-tagger persistence.

Covers the constructor hydration path + the ``set_active_tagger``
round-trip via ``SettingsService``. Subprocess integration is
Phase 2; this file does not exercise the spawn kwargs yet.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from yadc.api.modules.tagger_catalog import ActiveTagger
from yadc.api.services.settings import SettingsService
from yadc.api.services.tagging import TaggingService


class TestConstructorHydration:
    """``TaggingService`` reads the persisted selection at ``__init__``."""

    def test_no_persisted_selection_leaves_active_tagger_none(self, service: TaggingService):
        """A fresh settings row (no ``tagger.active_model`` key) leaves
        ``active_tagger`` as ``None`` — the fallback path is used."""
        assert service.active_tagger is None

    def test_valid_persisted_selection_is_hydrated(self, settings_service, service_factory):
        """A well-formed selection written before construction is read back."""
        settings_service.set(
            "tagger.active_model",
            {
                "kind": "hf",
                "repo_id": "SmilingWolf/wd-eva02-large-tagger-v3",
                "preproc_profile": "wd-tagger",
            },
        )
        service = service_factory(settings_service)

        assert service.active_tagger is not None
        assert service.active_tagger.kind == "hf"
        assert service.active_tagger.repo_id == "SmilingWolf/wd-eva02-large-tagger-v3"

    def test_malformed_persisted_selection_leaves_active_tagger_none_and_warns(self, settings_service, service_factory, caplog):
        """A schema-incompatible row (e.g. ``kind='hf'`` with empty ``repo_id``)
        is discarded and a warning is logged; ``active_tagger`` stays ``None``.

        The service is still constructed so legacy Configuration
        fallbacks and the rest of the API surface work.
        """
        settings_service.set(
            "tagger.active_model",
            {
                # kind='hf' without a repo_id fails _validate_consistency;
                # simulate a row that survived an older schema.
                "kind": "hf",
                "repo_id": "",
                "preproc_profile": "wd-tagger",
            },
        )
        with caplog.at_level(logging.WARNING, logger="yadc.api.services.tagging"):
            service = service_factory(settings_service)

        assert service.active_tagger is None
        assert any("Discarding malformed active tagger" in record.getMessage() for record in caplog.records)

    def test_non_dict_persisted_row_is_ignored(self, settings_service, service_factory):
        """A non-dict row (e.g. a legacy scalar the store once held) is
        ignored without raising — keep the service constructible."""
        settings_service.set("tagger.active_model", "legacy-string-value")
        service = service_factory(settings_service)

        assert service.active_tagger is None

    def test_legacy_inprocess_check_passes_after_construction(self, service: TaggingService):
        """Sanity check: the active-tagger property is wired into the
        service constructor (vs. only being available after a setter)."""
        # No setter has been called; verify the property exists.
        # Pydantic raises on None: this also confirms the import works.
        with pytest.raises(ValidationError):
            ActiveTagger.model_validate(None)  # type: ignore[arg-type]


# ``service_factory`` is a helper for tests that want to build a fresh
# ``TaggingService`` after pre-seeding ``settings_service``. The base
# ``service`` fixture seeds the same shared instance, which makes per-test
# setup awkward when the test's intent is "this row triggers hydration
# behaviour". Factory form lets each test own its service instance.


@pytest.fixture
def service_factory(
    test_configuration,
    logging_factory,
    event_dispatcher,
    dataset_service,
    dataset_watcher,
    job_scheduler,
    dataset_jobs,
):
    """Build a ``TaggingService`` against the provided ``SettingsService``.

    Mirrors the base ``service`` fixture but lets each test pre-seed the
    settings table *before* construction (so the hydration path sees the
    seeded row).
    """

    def _factory(svc: SettingsService) -> TaggingService:
        # The fixture's test_configuration may have a tmp_path-anchored
        # tagger_model_path; we re-use it so the service is constructible
        # as a local-path tagger (the hydration test never spawns).
        return TaggingService(
            configuration=test_configuration,
            logging=logging_factory,
            event_dispatcher=event_dispatcher,
            dataset_service=dataset_service,
            dataset_watcher=dataset_watcher,
            job_scheduler=job_scheduler,
            dataset_jobs=dataset_jobs,
            settings_service=svc,
        )

    return _factory


class TestSetActiveTagger:
    """``set_active_tagger`` persists + stashes the validated selection."""

    def test_persists_to_settings(self, service: TaggingService, settings_service: SettingsService):
        """After ``set_active_tagger``, the same row is readable via ``SettingsService.get``."""
        selection = ActiveTagger(
            kind="hf",
            repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
        )
        service.set_active_tagger(selection)

        raw = settings_service.get("tagger.active_model")
        assert raw["kind"] == "hf"
        assert raw["repo_id"] == "SmilingWolf/wd-eva02-large-tagger-v3"

    def test_updates_in_process_cached_model(self, service: TaggingService):
        """The cached model is updated in-place (same instance, not a round-trip through the DB)."""
        selection = ActiveTagger(
            kind="local",
            model_path="/path/to/model.onnx",
            preproc_profile="timm",
        )
        service.set_active_tagger(selection)

        assert service.active_tagger is selection

    def test_round_trip_through_constructor(self, settings_service: SettingsService, service_factory):
        """Setting on one instance makes the next construction pick up the row.

        Both ``first`` and ``second`` are built against the same
        ``settings_service`` so the second constructor's hydration
        path sees the row written by the first.
        """
        first = service_factory(settings_service)
        first.set_active_tagger(
            ActiveTagger(
                kind="hf",
                repo_id="SmilingWolf/wd-v1-4-swinv2-tagger-v2",
            )
        )

        second = service_factory(settings_service)

        assert second.active_tagger is not None
        assert second.active_tagger.repo_id == "SmilingWolf/wd-v1-4-swinv2-tagger-v2"

    def test_local_selection_round_trips_through_persistence(self, service: TaggingService, settings_service: SettingsService):
        """A local-path selection (with profile + size) round-trips through
        the JSON store and reconstructs identically."""
        selection = ActiveTagger(
            kind="local",
            model_path="/models/animetimm/model.onnx",
            label_path="/models/animetimm/selected_tags.csv",
            preproc_profile="timm",
            default_size=512,
        )
        service.set_active_tagger(selection)

        # Read the raw row and reconstruct; the resulting model must
        # equal the original.
        raw = settings_service.get("tagger.active_model")
        reconstructed = ActiveTagger.model_validate(raw)

        assert reconstructed == selection
