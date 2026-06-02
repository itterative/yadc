"""Filesystem watcher for dataset directories using watchdog.

Monitors all registered dataset image directories for changes (new files,
removed files, modified sidecars) and emits ``DatasetChangedEvent`` via the
``EventDispatcher``. Events are debounced per-dataset to coalesce rapid bursts.
"""

from __future__ import annotations

import threading
from logging import Logger
from pathlib import Path
from typing import Any, Callable, override

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..events import DatasetChangedEvent
from .event_dispatcher import EventDispatcher
from .logging_factory import LoggingFactory
from .service import Service

# Extensions we care about — both images and sidecars.
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".ico"})
_SIDECAR_EXTENSIONS = frozenset({".txt", ".toml", ".history~", ".draft~"})
_WATCHED_EXTENSIONS = _IMAGE_EXTENSIONS | _SIDECAR_EXTENSIONS


def _is_watched(path: str) -> bool:
    """Check if a file path has an extension we care about."""
    name = Path(path).name
    for ext in _WATCHED_EXTENSIONS:
        if name.endswith(ext):
            return True
    return False


class _DirEventHandler(FileSystemEventHandler):
    """Watchdog handler for a single dataset's watched directories."""

    def __init__(self, dataset_name: str, on_change: Callable[[str], None]):
        self._dataset_name: str = dataset_name
        self._on_change: Callable[[str], None] = on_change

    @override
    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_watched(str(event.src_path)):
            self._on_change(self._dataset_name)

    @override
    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_watched(str(event.src_path)):
            self._on_change(self._dataset_name)

    @override
    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and _is_watched(str(event.src_path)):
            self._on_change(self._dataset_name)

    @override
    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            if _is_watched(str(event.src_path)) or _is_watched(str(event.dest_path)):
                self._on_change(self._dataset_name)


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
    ):
        self._logger: Logger = logging.get_logger(__name__)
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self._debounce_seconds: float = 1.0  # set via set_debounce_seconds()

        self._observer: Any = Observer()
        self._observer.daemon = True

        # dataset_name -> list of (watch, handler) for cleanup
        self._watches: dict[str, list[tuple[object, _DirEventHandler]]] = {}
        # dataset_name -> debounce Timer
        self._timers: dict[str, threading.Timer] = {}
        self._lock: threading.Lock = threading.Lock()

        # Start the observer in a background thread
        self._thread: threading.Thread = threading.Thread(target=self._run_observer, daemon=True)
        self._thread.start()

    def _run_observer(self) -> None:
        """Run the watchdog observer (blocks until stop())."""
        self._observer.start()
        self._logger.info("Dataset watcher started. [debounce=%.1fs]", self._debounce_seconds)
        self._observer.join()

    def set_debounce_seconds(self, seconds: float) -> None:
        """Set the debounce interval. Should be called before any watch_dataset() calls."""
        self._debounce_seconds = seconds

    def stop(self) -> None:
        """Stop the filesystem observer."""
        self._observer.stop()

    def watch_dataset(self, dataset_name: str, paths: list[str]) -> None:
        """Add directories to watch for a dataset."""
        with self._lock:
            # Remove any existing watches for this dataset first
            self._unwatch_dataset_locked(dataset_name)

            handler = _DirEventHandler(dataset_name, self._on_fs_change)
            watches: list[tuple[object, _DirEventHandler]] = []
            for path_str in paths:
                path = Path(path_str)
                if not path.is_dir():
                    self._logger.warning("Watch path is not a directory, skipping. [path=%s]", path_str)
                    continue

                self._logger.debug("Watching dataset directory. [dataset=%s, path=%s]", dataset_name, path_str)
                watch = self._observer.schedule(handler, str(path), recursive=False)
                watches.append((watch, handler))

            self._watches[dataset_name] = watches

    def unwatch_dataset(self, dataset_name: str) -> None:
        """Remove all watched directories for a dataset."""
        with self._lock:
            self._unwatch_dataset_locked(dataset_name)

    def _unwatch_dataset_locked(self, dataset_name: str) -> None:
        """Remove watches for a dataset (caller must hold self._lock)."""
        # Cancel any pending debounce timer
        timer = self._timers.pop(dataset_name, None)
        if timer is not None:
            timer.cancel()

        watches = self._watches.pop(dataset_name, [])
        for watch, _handler in watches:
            try:
                self._observer.unschedule(watch)
            except Exception:
                pass  # observer may already be stopped

    def _on_fs_change(self, dataset_name: str) -> None:
        """Debounced callback — called by _DirEventHandler on any relevant change."""
        with self._lock:
            # Cancel existing timer
            old_timer = self._timers.pop(dataset_name, None)
            if old_timer is not None:
                old_timer.cancel()

            # Schedule a new one
            timer = threading.Timer(self._debounce_seconds, self._dispatch_change, args=(dataset_name,))
            timer.daemon = True
            self._timers[dataset_name] = timer

        timer.start()

    def _dispatch_change(self, dataset_name: str) -> None:
        """Dispatch a DatasetChangedEvent (called from debounce timer thread)."""
        with self._lock:
            self._timers.pop(dataset_name, None)

        self._logger.debug("Dispatching dataset_changed event. [dataset=%s]", dataset_name)
        self._event_dispatcher.dispatch(DatasetChangedEvent(dataset_name=dataset_name))
