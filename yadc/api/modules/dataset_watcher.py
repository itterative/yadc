"""Filesystem watcher for dataset directories using watchdog.

Monitors all registered dataset image directories for changes (new files,
removed files, modified sidecars) and emits ``DatasetChangedEvent`` via the
``EventDispatcher``. Events are debounced per-dataset to coalesce rapid bursts.
"""

from __future__ import annotations

import fnmatch
import threading
import time
from collections import deque
from logging import Logger
from pathlib import Path
from typing import Any, Callable, NamedTuple, override

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch

from ..configuration import Configuration
from ..events import DatasetChangedEvent, StartupEvent
from .event_dispatcher import EventDispatcher, event_handler
from .logging_factory import LoggingFactory
from .service import Service

# Sentinel job_id used when all changes in a debounce window were expected
# but no client-specific source is available.
SELF_JOB_ID = "self"

# Prefix for frontend client sources (e.g. "ui:abc123").
UI_SOURCE_PREFIX = "ui:"


class ExpectedFileEntry(NamedTuple):
    path: str
    registered_at: float
    source: str


class ExpectedPatternEntry(NamedTuple):
    pattern: str
    registered_at: float
    source: str


# Extensions we care about — both images and sidecars.
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".ico"})
SIDECAR_EXTENSIONS: frozenset[str] = frozenset({".txt", ".toml", ".history~", ".draft~"})
# Glob suffixes for sidecars when attached to a stem (used in expect_pattern_change).
# .txt / .toml / .history~ are exact-match patterns; .*.draft~ matches named drafts.
SIDECAR_EXTENSION_GLOBS: tuple[str, ...] = (".txt", ".toml", ".history~", ".*.draft~")
_WATCHED_EXTENSIONS = _IMAGE_EXTENSIONS | SIDECAR_EXTENSIONS


def _is_watched(path: str) -> bool:
    """Check if a file path has an extension we care about."""
    name = Path(path).name
    for ext in _WATCHED_EXTENSIONS:
        if name.endswith(ext):
            return True
    return False


class _DirEventHandler(FileSystemEventHandler):
    """Watchdog handler for a single dataset's watched directories."""

    def __init__(self, dataset_name: str, on_change: Callable[[str, str], None]):
        self._dataset_name: str = dataset_name
        self._on_change: Callable[[str, str], None] = on_change

    @override
    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_watched(str(event.src_path)):
            self._on_change(self._dataset_name, str(event.src_path))

    @override
    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_watched(str(event.src_path)):
            self._on_change(self._dataset_name, str(event.src_path))

    @override
    def on_closed(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_watched(str(event.src_path)):
            self._on_change(self._dataset_name, str(event.src_path))

    @override
    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            if _is_watched(str(event.src_path)) or _is_watched(str(event.dest_path)):
                self._on_change(self._dataset_name, str(event.src_path))


class DatasetWatcherService(Service):
    """Watches dataset directories for filesystem changes and emits SSE events.

    One ``Observer`` instance is shared across all datasets. Each dataset's
    directories get their own ``_DirEventHandler`` that debounces changes and
    dispatches ``DatasetChangedEvent``.
    """

    def __init__(
        self,
        event_dispatcher: EventDispatcher,
        logging: LoggingFactory,
        configuration: Configuration,
    ):
        self._logger: Logger = logging.get_logger(__name__)
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self._configuration: Configuration = configuration

        self._observer: Any = Observer()
        self._observer.daemon = True

        # dataset_name -> list of (ObservedWatch, handler) for cleanup
        self._watches: dict[str, list[tuple[ObservedWatch, _DirEventHandler]]] = {}
        # dataset_name -> debounce Timer
        self._timers: dict[str, threading.Timer] = {}
        # dataset_name -> job_id for expected changes (captioning jobs)
        self._expected_sources: dict[str, str] = {}
        # dataset_name -> bounded deque of ExpectedFileEntry for webui edits
        self._expected_files: dict[str, deque[ExpectedFileEntry]] = {}
        # dataset_name -> bounded deque of ExpectedPatternEntry for bulk deletions
        self._expected_patterns: dict[str, deque[ExpectedPatternEntry]] = {}
        # dataset_name -> True if any unexpected file changed during debounce window
        self._unexpected_changes: dict[str, bool] = {}
        self._lock: threading.Lock = threading.Lock()

        self._thread: threading.Thread = threading.Thread(target=self._run_observer, daemon=True)

    @event_handler(StartupEvent)
    def on_startup(self, event: StartupEvent):  # pyright: ignore[reportUnusedParameter]
        # Start the observer in a background thread
        self._thread.start()

    def _run_observer(self) -> None:
        """Run the watchdog observer (blocks until stop())."""
        self._observer.start()
        self._logger.info("Dataset watcher started. [debounce=%.1fs]", self._configuration.watcher_debounce_seconds)
        self._observer.join()

    def stop(self) -> None:
        """Stop the filesystem observer."""
        self._observer.stop()

    def watch_dataset(self, dataset_name: str, paths: list[str]) -> None:
        """Add directories to watch for a dataset.

        If the watched paths haven't changed since the last call, this is a
        no-op — avoiding unnecessary unwatch/rewatch cycles that would disrupt
        ``_expected_sources`` and ``_expected_files`` state.
        """
        with self._lock:
            normalized = sorted(str(Path(p).resolve()) for p in paths)

            # Short-circuit if paths haven't changed
            current_watches = self._watches.get(dataset_name)
            if current_watches is not None:
                current_paths = sorted(str(Path(watch.path).resolve()) for watch, _handler in current_watches)
                if normalized == current_paths:
                    self._logger.debug("Watch paths unchanged, skipping re-registration. [dataset=%s]", dataset_name)
                    return

            # Preserve the expected-changes tag across re-registration
            # (e.g. when _refresh_stale_datasets picks up a path change during
            # captioning).  _unwatch_dataset_locked() clears it, but captioning
            # jobs expect it to remain active until explicitly cleared.
            saved_source = self._expected_sources.get(dataset_name)

            # Remove any existing watches for this dataset first
            self._unwatch_dataset_locked(dataset_name)

            handler = _DirEventHandler(dataset_name, self._on_fs_change)
            watches: list[tuple[ObservedWatch, _DirEventHandler]] = []
            for path_str in paths:
                path = Path(path_str)
                if not path.is_dir():
                    self._logger.warning("Watch path is not a directory, skipping. [path=%s]", path_str)
                    continue

                self._logger.debug("Watching dataset directory. [dataset=%s, path=%s]", dataset_name, path_str)
                watch = self._observer.schedule(handler, str(path), recursive=False)
                watches.append((watch, handler))

            self._watches[dataset_name] = watches

            # Restore the expected-changes tag if one was active
            if saved_source is not None:
                self._expected_sources[dataset_name] = saved_source

    def unwatch_dataset(self, dataset_name: str) -> None:
        """Remove all watched directories for a dataset."""
        with self._lock:
            self._unwatch_dataset_locked(dataset_name)

    def expect_changes(self, dataset_name: str, job_id: str) -> None:
        """Tag filesystem change events for *dataset_name* with *job_id* until cleared.

        Called by captioning to mark that DatasetChangedEvents
        are caused by a specific captioning job (so the initiating frontend can ignore them).
        Remains active until ``clear_expected_changes()`` is called.
        """
        with self._lock:
            self._expected_sources[dataset_name] = job_id

    def clear_expected_changes(self, dataset_name: str) -> None:
        """Remove the source tag for a dataset's change events."""
        with self._lock:
            self._expected_sources.pop(dataset_name, None)

    def clear_expected_changes_for_job(self, dataset_name: str, job_id: str) -> None:
        """Remove the source tag only if it still matches *job_id*.

        Prevents a stale cleanup timer from wiping a newer job's source tag.
        """
        with self._lock:
            if self._expected_sources.get(dataset_name) == job_id:
                self._expected_sources.pop(dataset_name, None)

    def expect_file_change(self, dataset_name: str, file_path: str, *, source: str = SELF_JOB_ID) -> None:
        """Register *file_path* as an expected upcoming filesystem change.

        Call before writing to a file (caption, extras, history) so the
        watcher can suppress the resulting ``DatasetChangedEvent`` when
        *all* changes in a debounce window are expected.  Entries expire
        after ``_expected_file_ttl`` seconds to handle duplicate inotify
        events for the same write.

        *source* is a client identifier included on the dispatched event so
        only the originating frontend tab suppresses it.  Defaults to
        ``SELF_JOB_ID`` (``"self"``) for backward compatibility.  Frontend
        clients should pass a ``"ui:<uuid>"`` string.
        """
        with self._lock:
            buf = self._expected_files.setdefault(dataset_name, deque(maxlen=self._configuration.watcher_expected_file_max))
            buf.append(ExpectedFileEntry(file_path, time.monotonic(), source))
            self._logger.debug(
                "Registered expected file change. [dataset=%s, path=%s, source=%s, queue=%d]",
                dataset_name,
                Path(file_path).name,
                source,
                len(buf),
            )

    def expect_pattern_change(self, dataset_name: str, pattern: str, *, source: str = SELF_JOB_ID) -> None:
        """Register a glob *pattern* as an expected upcoming filesystem change.

        Use before bulk deletions (entire folders, or draft files that
        match ``STEM.*.draft~``) where listing every individual file
        would be impractical.  Entries share the same TTL as
        :meth:`expect_file_change` and the same bounded deque limit.

        *pattern* is a ``fnmatch`` glob such as ``"/path/*.txt"`` or
        ``"/path/folders/train/*"``.
        """
        with self._lock:
            buf = self._expected_patterns.setdefault(dataset_name, deque(maxlen=self._configuration.watcher_expected_file_max))
            buf.append(ExpectedPatternEntry(pattern, time.monotonic(), source))
            self._logger.debug(
                "Registered expected pattern change. [dataset=%s, pattern=%s, source=%s, queue=%d]",
                dataset_name,
                pattern,
                source,
                len(buf),
            )

    def _unwatch_dataset_locked(self, dataset_name: str) -> None:
        """Remove watches for a dataset (caller must hold self._lock)."""
        assert self._lock.locked(), "_unwatch_dataset_locked must be called with self._lock held"

        # Cancel any pending debounce timer
        timer = self._timers.pop(dataset_name, None)
        if timer is not None:
            timer.cancel()

        self._expected_sources.pop(dataset_name, None)
        self._expected_files.pop(dataset_name, None)
        self._expected_patterns.pop(dataset_name, None)
        self._unexpected_changes.pop(dataset_name, None)

        watches = self._watches.pop(dataset_name, [])
        for watch, _handler in watches:
            try:
                self._observer.unschedule(watch)
            except Exception:
                pass  # observer may already be stopped

    def _on_fs_change(self, dataset_name: str, file_path: str) -> None:
        """Debounced callback — called by _DirEventHandler on any relevant change."""
        with self._lock:
            # Check if this file was expected (within TTL).  We do NOT consume
            # the entry on match — a single write can produce multiple inotify
            # events, so the entry must remain valid for the TTL window.
            now = time.monotonic()

            # Exact path match
            expected = self._expected_files.get(dataset_name)
            matched = False
            if expected:
                for entry in expected:
                    if entry.path == file_path and (now - entry.registered_at) < self._configuration.watcher_expected_file_ttl:
                        matched = True
                        break

            # Pattern match (fnmatch glob)
            if not matched:
                patterns = self._expected_patterns.get(dataset_name)
                if patterns:
                    for entry in patterns:
                        if fnmatch.fnmatch(file_path, entry.pattern) and (now - entry.registered_at) < self._configuration.watcher_expected_file_ttl:
                            matched = True
                            break

            if matched:
                self._logger.debug(
                    "Expected file change matched. [dataset=%s, path=%s, queue=%d]",
                    dataset_name,
                    Path(file_path).name,
                    len(expected) if expected else 0,
                )
            else:
                # At least one unexpected change in this debounce window
                was_unexpected = self._unexpected_changes.get(dataset_name, False)
                self._unexpected_changes[dataset_name] = True
                self._logger.debug(
                    "Unexpected file change. [dataset=%s, path=%s, had_prior_unexpected=%s]",
                    dataset_name,
                    Path(file_path).name,
                    was_unexpected,
                )

            # Cancel existing timer (debounce restart)
            old_timer = self._timers.pop(dataset_name, None)
            is_rebatch = old_timer is not None
            if old_timer is not None:
                old_timer.cancel()

            # Capture the job_id now so a later cleanup can't wipe it before the
            # debounce fires. If no job is active, job_id will be None.
            job_id = self._expected_sources.get(dataset_name)

            # Schedule a new one
            timer = threading.Timer(self._configuration.watcher_debounce_seconds, self._dispatch_change, args=(dataset_name, job_id))
            timer.daemon = True
            self._timers[dataset_name] = timer

            if is_rebatch:
                self._logger.debug(
                    "Debounce rebatched. [dataset=%s, job_id=%s, unexpected=%s]",
                    dataset_name,
                    job_id,
                    self._unexpected_changes.get(dataset_name, False),
                )

        timer.start()

    def _dispatch_change(self, dataset_name: str, job_id: str | None) -> None:
        """Dispatch a DatasetChangedEvent (called from debounce timer thread)."""
        with self._lock:
            self._timers.pop(dataset_name, None)
            unexpected = self._unexpected_changes.pop(dataset_name, False)
            expected_entries = self._expected_files.pop(dataset_name, deque())
            expected_patterns = self._expected_patterns.pop(dataset_name, deque())

        # If all changes in this debounce window were expected (no unexpected
        # changes) and no captioning job_id, derive a source from the expected
        # file entries and patterns so only the originating client tab suppresses.
        suppressed = False
        if not unexpected and job_id is None:
            # Collect unique sources from both exact entries and patterns.
            # If all share the same source, use it; otherwise fall back to SELF_JOB_ID.
            sources = {entry.source for entry in expected_entries} | {entry.source for entry in expected_patterns}
            job_id = sources.pop() if len(sources) == 1 else SELF_JOB_ID
            suppressed = True

        self._logger.debug(
            "Dispatching dataset_changed. [dataset=%s, job_id=%s, unexpected=%s, expected=%d, patterns=%d, suppressed=%s]",
            dataset_name,
            job_id,
            unexpected,
            len(expected_entries),
            len(expected_patterns),
            suppressed,
        )
        self._event_dispatcher.dispatch(DatasetChangedEvent(dataset_name=dataset_name, job_id=job_id))
