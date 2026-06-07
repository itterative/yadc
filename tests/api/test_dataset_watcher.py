"""Tests for DatasetWatcherService — path-diff short-circuit and expected-sources preservation."""

import time
from collections import deque
from unittest.mock import MagicMock

import pytest

from yadc.api.events import DatasetChangedEvent
from yadc.api.modules.dataset_watcher import SELF_JOB_ID, DatasetWatcherService, ExpectedFileEntry, ExpectedPatternEntry
from yadc.api.modules.thread_factory import ThreadFactory


@pytest.fixture
def watcher(test_configuration, logging_factory):
    """DatasetWatcherService with an injected mock observer and thread factory.

    The observer is the only dependency the tests want to control —
    every test asserts on ``_observer.schedule`` / ``unschedule`` calls.
    The constructor doesn't start the underlying watchdog thread (that
    happens in ``on_startup``, which these tests never invoke), so the
    other constructor params can be cheap stubs. ``ThreadFactory`` is
    injected (not created internally) so the watcher doesn't create
    threads just by being constructed.
    """
    return DatasetWatcherService(
        event_dispatcher=MagicMock(),
        logging=logging_factory,
        configuration=test_configuration,
        observer=MagicMock(),
        thread_factory=MagicMock(spec=ThreadFactory),
    )


@pytest.fixture
def dirs(tmp_path):
    """Create two temporary directories to use as watch paths."""
    d1 = tmp_path / "images"
    d1.mkdir()
    d2 = tmp_path / "extras"
    d2.mkdir()
    return [str(d1), str(d2)]


class TestWatchDatasetPathDiff:
    """Tests for watch_dataset short-circuiting when paths haven't changed."""

    def test_schedule_called_on_first_watch(self, watcher, dirs):
        """First call should schedule all paths."""
        watcher._observer.schedule.return_value = MagicMock(path=dirs[0])

        watcher.watch_dataset("ds", dirs)

        assert watcher._observer.schedule.call_count == 2
        assert "ds" in watcher._watches

    def test_short_circuit_on_same_paths(self, watcher, dirs):
        """Second call with identical paths should not re-schedule."""
        # Create mock watches that have a .path attribute
        mock_watch_0 = MagicMock(path=dirs[0])
        mock_watch_1 = MagicMock(path=dirs[1])
        watcher._observer.schedule.side_effect = [mock_watch_0, mock_watch_1]

        watcher.watch_dataset("ds", dirs)
        assert watcher._observer.schedule.call_count == 2

        # Reset so we can detect new calls
        watcher._observer.schedule.reset_mock()

        # Call again with same paths
        watcher.watch_dataset("ds", dirs)

        watcher._observer.schedule.assert_not_called()

    def test_short_circuit_preserves_expected_sources(self, watcher, dirs):
        """Short-circuit should not touch _expected_sources."""
        watcher._observer.schedule.return_value = MagicMock(path=dirs[0])

        # Set up initial watch
        watcher.watch_dataset("ds", dirs)

        # Simulate captioning job tagging
        watcher.expect_changes("ds", "job_123")
        assert watcher._expected_sources["ds"] == "job_123"

        # Re-watch with same paths (short-circuits)
        watcher.watch_dataset("ds", dirs)

        assert watcher._expected_sources["ds"] == "job_123"

    def test_reregister_on_changed_paths(self, watcher, tmp_path):
        """Paths changed → full unwatch + rewatch should happen."""
        old_dir = tmp_path / "old"
        old_dir.mkdir()
        new_dir = tmp_path / "new"
        new_dir.mkdir()

        watcher._observer.schedule.return_value = MagicMock(path=str(old_dir))

        watcher.watch_dataset("ds", [str(old_dir)])
        assert watcher._observer.schedule.call_count == 1

        # Reset
        watcher._observer.schedule.reset_mock()
        watcher._observer.schedule.return_value = MagicMock(path=str(new_dir))

        # Watch with different path
        watcher.watch_dataset("ds", [str(new_dir)])

        watcher._observer.schedule.assert_called_once()

    def test_reregister_preserves_expected_sources(self, watcher, tmp_path):
        """When paths change, _expected_sources should be saved and restored."""
        old_dir = tmp_path / "old"
        old_dir.mkdir()
        new_dir = tmp_path / "new"
        new_dir.mkdir()

        watcher._observer.schedule.return_value = MagicMock(path=str(old_dir))
        watcher.watch_dataset("ds", [str(old_dir)])

        # Simulate captioning job tagging
        watcher.expect_changes("ds", "job_456")

        # Re-watch with different path
        watcher._observer.schedule.return_value = MagicMock(path=str(new_dir))
        watcher.watch_dataset("ds", [str(new_dir)])

        assert watcher._expected_sources["ds"] == "job_456"

    def test_unwatch_clears_expected_sources(self, watcher, dirs):
        """unwatch_dataset should clear _expected_sources."""
        watcher._observer.schedule.return_value = MagicMock(path=dirs[0])
        watcher.watch_dataset("ds", dirs)
        watcher.expect_changes("ds", "job_789")

        watcher.unwatch_dataset("ds")

        assert "ds" not in watcher._expected_sources

    def test_path_ordering_doesnt_matter(self, watcher, dirs):
        """Paths in different order should still short-circuit."""
        # Mock schedule to return watches with the correct path for each call
        watcher._observer.schedule.side_effect = [MagicMock(path=dirs[0]), MagicMock(path=dirs[1])]
        watcher.watch_dataset("ds", dirs)

        watcher._observer.schedule.reset_mock()

        # Reverse order — same normalized paths, should short-circuit
        watcher.watch_dataset("ds", list(reversed(dirs)))

        watcher._observer.schedule.assert_not_called()

    def test_empty_paths_no_prior_watches(self, watcher, dirs):
        """Calling with empty paths when no watches exist creates an empty watch list."""
        watcher.watch_dataset("ds", [])
        assert watcher._watches["ds"] == []
        watcher._observer.schedule.assert_not_called()


class TestWatchDatasetUnwatchResetsState:
    """Tests for _unwatch_dataset_locked clearing all per-dataset state."""

    def test_unwatch_clears_timers(self, watcher, dirs):
        watcher._observer.schedule.return_value = MagicMock(path=dirs[0])
        watcher.watch_dataset("ds", dirs)

        # Simulate a pending timer
        mock_timer = MagicMock()
        watcher._timers["ds"] = mock_timer

        watcher.unwatch_dataset("ds")

        mock_timer.cancel.assert_called_once()
        assert "ds" not in watcher._timers

    def test_unwatch_unschedules_observer(self, watcher, dirs):
        mock_watch = MagicMock(path=dirs[0])
        watcher._observer.schedule.return_value = mock_watch
        watcher.watch_dataset("ds", dirs)

        watcher.unwatch_dataset("ds")

        watcher._observer.unschedule.assert_called_with(mock_watch)
        assert "ds" not in watcher._watches


class TestDispatchChangeSource:
    """Tests for _dispatch_change deriving source from expected file entries."""

    def test_uniform_pattern_source_uses_it(self, watcher):
        """When all expected patterns share the same source, that source is used."""
        watcher._expected_patterns["ds"] = [
            ExpectedPatternEntry("/a/*.txt", 0, "ui:abc"),
        ]
        watcher._unexpected_changes["ds"] = False

        watcher._dispatch_change("ds", None)

        event = watcher._event_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, DatasetChangedEvent)
        assert event.job_id == "ui:abc"

    def test_mixed_pattern_and_file_sources_falls_back_to_self(self, watcher):
        """When expected files and patterns have different sources, falls back to 'self'."""
        watcher._expected_files["ds"] = [
            ExpectedFileEntry("/a.txt", 0, "ui:abc"),
        ]
        watcher._expected_patterns["ds"] = [
            ExpectedPatternEntry("/b/*.txt", 0, "ui:def"),
        ]
        watcher._unexpected_changes["ds"] = False

        watcher._dispatch_change("ds", None)

        event = watcher._event_dispatcher.dispatch.call_args[0][0]
        assert event.job_id == SELF_JOB_ID

    def test_no_expected_patterns_no_job_id_dispatches_none(self, watcher):
        """No expected patterns/files but unexpected changes → dispatches with None."""
        watcher._unexpected_changes["ds"] = True

        watcher._dispatch_change("ds", None)

        event = watcher._event_dispatcher.dispatch.call_args[0][0]
        assert event.job_id is None

    def test_uniform_source_uses_it(self, watcher):
        """When all expected entries share the same source, that source is used."""
        watcher._expected_files["ds"] = [
            ExpectedFileEntry("/a.txt", 0, "ui:abc"),
            ExpectedFileEntry("/a.toml", 0, "ui:abc"),
        ]
        watcher._unexpected_changes["ds"] = False

        watcher._dispatch_change("ds", None)

        event = watcher._event_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, DatasetChangedEvent)
        assert event.job_id == "ui:abc"

    def test_mixed_sources_falls_back_to_self(self, watcher):
        """When expected entries have different sources, falls back to SELF_JOB_ID."""
        watcher._expected_files["ds"] = [
            ExpectedFileEntry("/a.txt", 0, "ui:abc"),
            ExpectedFileEntry("/b.txt", 0, "ui:def"),
        ]
        watcher._unexpected_changes["ds"] = False

        watcher._dispatch_change("ds", None)

        event = watcher._event_dispatcher.dispatch.call_args[0][0]
        assert event.job_id == SELF_JOB_ID

    def test_default_source_uses_self(self, watcher):
        """Entries with default source (SELF_JOB_ID) use 'self'."""
        watcher._expected_files["ds"] = [
            ExpectedFileEntry("/a.txt", 0, SELF_JOB_ID),
        ]
        watcher._unexpected_changes["ds"] = False

        watcher._dispatch_change("ds", None)

        event = watcher._event_dispatcher.dispatch.call_args[0][0]
        assert event.job_id == "self"

    def test_no_expected_files_no_job_id_dispatches_none(self, watcher):
        """No expected files but unexpected changes → dispatches with None."""
        watcher._unexpected_changes["ds"] = True

        watcher._dispatch_change("ds", None)

        event = watcher._event_dispatcher.dispatch.call_args[0][0]
        assert event.job_id is None

    def test_captioning_job_id_takes_priority(self, watcher):
        """Captioning job_id from _expected_sources takes priority over file sources."""
        watcher._expected_files["ds"] = [
            ExpectedFileEntry("/a.txt", 0, "ui:abc"),
        ]
        watcher._unexpected_changes["ds"] = False

        watcher._dispatch_change("ds", "captioning_job_123")

        event = watcher._event_dispatcher.dispatch.call_args[0][0]
        assert event.job_id == "captioning_job_123"

    def test_captioning_job_id_takes_priority_over_patterns(self, watcher):
        """Captioning job_id takes priority even when patterns are present."""
        watcher._expected_patterns["ds"] = [
            ExpectedPatternEntry("/a/*.txt", 0, "ui:abc"),
        ]
        watcher._unexpected_changes["ds"] = False

        watcher._dispatch_change("ds", "captioning_job_123")

        event = watcher._event_dispatcher.dispatch.call_args[0][0]
        assert event.job_id == "captioning_job_123"


class TestOnFsChangePatternMatching:
    """Tests for _on_fs_change matching against ExpectedPatternEntry."""

    def test_exact_match_still_works(self, watcher):
        """Exact file entries should still match as before."""
        now = time.monotonic()
        watcher._expected_files["ds"] = [
            ExpectedFileEntry("/img/a.txt", now, SELF_JOB_ID),
        ]
        watcher._expected_patterns["ds"] = deque()

        watcher._on_fs_change("ds", "/img/a.txt")

        assert watcher._unexpected_changes.get("ds") is not True

    def test_pattern_match(self, watcher):
        """A file path matching a registered glob pattern should be treated as expected."""
        now = time.monotonic()
        watcher._expected_files["ds"] = deque()
        watcher._expected_patterns["ds"] = [
            ExpectedPatternEntry("/img/a.*.draft~", now, SELF_JOB_ID),
        ]

        watcher._on_fs_change("ds", "/img/a.gemma.draft~")

        assert watcher._unexpected_changes.get("ds") is not True

    def test_pattern_no_match(self, watcher):
        """A file path NOT matching a registered glob pattern should be unexpected."""
        now = time.monotonic()
        watcher._expected_files["ds"] = deque()
        watcher._expected_patterns["ds"] = [
            ExpectedPatternEntry("/img/a.*.draft~", now, SELF_JOB_ID),
        ]

        watcher._on_fs_change("ds", "/img/b.gemma.draft~")

        assert watcher._unexpected_changes.get("ds") is True

    def test_folder_deletion_pattern(self, watcher):
        """Pattern for 'folder/*' should match any file inside the folder."""
        now = time.monotonic()
        watcher._expected_files["ds"] = deque()
        watcher._expected_patterns["ds"] = [
            ExpectedPatternEntry("/datasets/ds/folders/train/*", now, SELF_JOB_ID),
        ]

        watcher._on_fs_change("ds", "/datasets/ds/folders/train/cat.jpg")
        watcher._on_fs_change("ds", "/datasets/ds/folders/train/cat.txt")

        assert watcher._unexpected_changes.get("ds") is not True

    def test_pattern_ttl_expiry(self, watcher):
        """Expired patterns (older than watcher_expected_file_ttl) should not match."""
        watcher._expected_files["ds"] = deque()
        # Registered long ago — will be expired
        watcher._expected_patterns["ds"] = [
            ExpectedPatternEntry("/img/*.txt", time.monotonic() - 100.0, SELF_JOB_ID),
        ]

        watcher._on_fs_change("ds", "/img/a.txt")

        assert watcher._unexpected_changes.get("ds") is True

    def test_both_exact_and_pattern_mixed(self, watcher):
        """A mix of exact entries and patterns should both suppress events."""
        now = time.monotonic()
        watcher._expected_files["ds"] = [
            ExpectedFileEntry("/img/a.txt", now, "ui:abc"),
        ]
        watcher._expected_patterns["ds"] = [
            ExpectedPatternEntry("/img/a.*.draft~", now, "ui:abc"),
        ]

        watcher._on_fs_change("ds", "/img/a.txt")
        watcher._on_fs_change("ds", "/img/a.gemma.draft~")
        # b.txt is unexpected
        watcher._on_fs_change("ds", "/img/b.txt")

        assert watcher._unexpected_changes.get("ds") is True


class TestUnwatchPatterns:
    """Tests that unwatch_dataset clears expected patterns."""

    def test_unwatch_clears_patterns(self, watcher, dirs):
        watcher._observer.schedule.return_value = MagicMock(path=dirs[0])
        watcher.watch_dataset("ds", dirs)
        watcher._expected_patterns["ds"] = deque([ExpectedPatternEntry("/img/*", 0, SELF_JOB_ID)])

        watcher.unwatch_dataset("ds")

        assert "ds" not in watcher._expected_patterns
