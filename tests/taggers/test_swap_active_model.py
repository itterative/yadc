"""Tests for :meth:`TaggingService.swap_active_model`.

Covers:

- Happy path: persists, updates in-process, returns selection.
- Refuses (409) when a batch job is running.
- Refuses (429) when another swap is already in flight.
- Respawn happens (mocked subprocess via ``make_client_mock`` /
  ``patch_client_factory`` from conftest).
- Rollback semantics on respawn failure.
- Persist-after-respawn: failure logs but doesn't roll back.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from yadc.api.configuration import Configuration
from yadc.api.modules.tagger_catalog import ActiveTagger, ActiveTaggerKind
from yadc.api.services.settings import SettingsService
from yadc.api.services.tagging import (
    TaggerBusyError,
    TaggerSwapInProgressError,
    TaggingService,
)

from .conftest import make_client_mock, patch_client_factory, run_swap


def _selection(repo_id: str = "SmilingWolf/wd-vit-tagger-v3", *, kind: ActiveTaggerKind = "hf") -> ActiveTagger:
    return ActiveTagger(
        kind=kind,
        repo_id=repo_id,
        preproc_profile="wd-tagger",
    )


class TestSwapHappyPath:
    """A swap that proceeds end-to-end (mocked subprocess)."""

    def test_persists_active_model(self, service: TaggingService, settings_service: SettingsService):
        """The persisted settings row reflects the swap so a restart loads the new selection."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            run_swap(service, _selection())

        raw = settings_service.get("tagger.active_model")
        assert raw["kind"] == "hf"
        assert raw["repo_id"] == "SmilingWolf/wd-vit-tagger-v3"

    def test_updates_in_process_cached_model(self, service: TaggingService):
        """``active_tagger`` reflects the swapped selection after the swap returns."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            run_swap(service, _selection())

        assert service.active_tagger is not None
        assert service.active_tagger.repo_id == "SmilingWolf/wd-vit-tagger-v3"

    def test_returns_the_persisted_selection(self, service: TaggingService):
        """The return value is the same ActiveTagger instance the caller passed."""
        client = make_client_mock(alive=True)
        selection = _selection()
        with patch_client_factory(client):
            result = run_swap(service, selection)

        assert result is selection

    def test_tears_down_old_subprocess_before_respawn(self, test_configuration: Configuration, service: TaggingService):
        """A running subprocess is stopped before the new one is spawned, so the
        user sees the full stopping → stopped → starting → ready transition."""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            # Spawn an initial subprocess via a no-op swap (with a different repo).
            # Cleaner: spawn via ``tag_image`` but that needs a real image fixture;
            # simplest is to call swap twice and observe the start count.
            run_swap(service, _selection("SmilingWolf/wd-eva02-large-tagger-v3"))
            client.start.reset_mock()

            run_swap(service, _selection("SmilingWolf/wd-vit-tagger-v3"))

        # Exactly one spawn per swap (the client mock is reused across swaps;
        # we reset between them).
        assert client.start.await_count == 1
        assert client.stop.await_count == 1


class TestSwapRefusals:
    """The two typed exceptions raised before any state mutation."""

    def test_refuses_with_TaggerBusyError_when_batch_running(self, service: TaggingService):
        """A running batch job makes the swap raise ``TaggerBusyError`` and the
        persisted selection is left untouched."""
        # Inject a synthetic "running" job into ``_tag_jobs``: a mock task
        # that always reports ``not done()``. No real coroutine is needed
        # because the busy check only inspects ``state.task.done()``.
        fake_task = MagicMock()
        fake_task.done.return_value = False
        service._tag_jobs["ds"] = MagicMock(task=fake_task)
        try:
            with pytest.raises(TaggerBusyError, match="batch tagging job is running"):
                run_swap(service, _selection())
        finally:
            service._tag_jobs.pop("ds", None)

    def test_refuses_with_TaggerSwapInProgressError_when_concurrent(self, service: TaggingService):
        """Manually holding the swap-in-progress lock simulates an in-flight swap;
        the second caller gets ``TaggerSwapInProgressError``."""
        acquired = service._swap_in_progress_lock.acquire(blocking=False)
        assert acquired, "test setup: lock should be free initially"
        try:
            with pytest.raises(TaggerSwapInProgressError, match="another swap") as excinfo:
                run_swap(service, _selection())
            assert excinfo.value.retry_after_s == 2.0
        finally:
            service._swap_in_progress_lock.release()

    def test_busy_refusal_does_not_persist_anything(self, service: TaggingService, settings_service: SettingsService):
        """A 409 refusal must not write to SettingsService — the user's existing
        selection (or absence of one) stays as-is."""
        fake_task = MagicMock()
        fake_task.done.return_value = False
        service._tag_jobs["ds"] = MagicMock(task=fake_task)
        try:
            with pytest.raises(TaggerBusyError):
                run_swap(service, _selection())
        finally:
            service._tag_jobs.pop("ds", None)

        assert settings_service.get("tagger.active_model") is None


class TestSwapRollback:
    """Failure paths should leave the service in a usable state."""

    def test_respawn_failure_rolls_back_active_tagger(self, test_configuration: Configuration, service: TaggingService):
        """When the respawn ``_ensure_running_locked`` raises, the background swap
        rolls ``_active_tagger`` back to its previous value. The failure is
        surfaced via the ``failed`` SSE event (not a raised exception) because
        the swap runs detached from the HTTP handler."""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            run_swap(service, _selection("SmilingWolf/wd-eva02-large-tagger-v3"))
            first_active = service.active_tagger

        with patch_client_factory(client), pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                service,
                "_ensure_running_locked",
                AsyncMock(side_effect=RuntimeError("boom")),
            )
            # No exception escapes: the background task swallows it and
            # dispatches a ``failed`` SSE event instead.
            run_swap(service, _selection("SmilingWolf/wd-vit-tagger-v3"))

        assert service.active_tagger is not None
        assert service.active_tagger == first_active
        assert service.active_tagger.repo_id == "SmilingWolf/wd-eva02-large-tagger-v3"


class TestSwapPersistFailure:
    """Persist-after-respawn failure logs but doesn't roll back."""

    def test_persist_failure_leaves_in_memory_state_intact(self, service: TaggingService):
        """A SettingsService write that raises is logged but the in-memory swap
        stands — the user gets the new model this session even though a restart
        would roll it back."""
        client = make_client_mock(alive=True)
        broken_settings = MagicMock(spec=SettingsService)
        broken_settings.get = MagicMock(return_value=None)
        broken_settings.set = MagicMock(side_effect=RuntimeError("disk full"))

        with patch_client_factory(client), pytest.MonkeyPatch.context() as mp:
            mp.setattr(service, "_settings_service", broken_settings)
            run_swap(service, _selection())

        assert service.active_tagger is not None
        assert service.active_tagger.repo_id == "SmilingWolf/wd-vit-tagger-v3"


class TestSwapNoOp:
    """Same-identity submissions short-circuit before any state mutation."""

    def test_same_selection_returns_without_respawn(self, service: TaggingService):
        """A second swap with the identical ActiveTagger returns the cached
        selection, doesn't touch the subprocess, and doesn't re-persist."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            run_swap(service, _selection())
            client.start.reset_mock()
            client.stop.reset_mock()

            result = run_swap(service, _selection())

        assert result is service.active_tagger
        assert client.start.await_count == 0
        assert client.stop.await_count == 0

    def test_same_selection_proceeds_when_active_tagger_is_none(self, test_configuration: Configuration, service: TaggingService):
        """When no swap has been persisted yet (``_active_tagger`` is None),
        a swap that matches the flat Configuration fallback is NOT a no-op
        — we proceed so the selection gets persisted into SettingsService
        (migration path from legacy Configuration to the new store)."""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
        test_configuration.tagger_preproc_profile = "wd-tagger"
        # ``_active_tagger`` is None at this point because no swap has run.
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            run_swap(service, _selection("SmilingWolf/wd-eva02-large-tagger-v3"))

        assert service.active_tagger is not None
        assert service.active_tagger.repo_id == "SmilingWolf/wd-eva02-large-tagger-v3"

    def test_different_profile_does_not_no_op(self, test_configuration: Configuration, service: TaggingService):
        """A profile change with the same repo is a real swap — the subprocess
        rebuilds with the new preproc kwargs, so the no-op check must compare
        all persisted fields, not just ``source_label``."""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
        test_configuration.tagger_preproc_profile = "wd-tagger"
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            run_swap(service, _selection("SmilingWolf/wd-eva02-large-tagger-v3"))
            client.start.reset_mock()

            selection = ActiveTagger(
                kind="hf",
                repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
                preproc_profile="timm",
            )
            run_swap(service, selection)

        assert client.start.await_count == 1
        assert client.stop.await_count == 1
        assert service.active_tagger is not None
        assert service.active_tagger.preproc_profile == "timm"
