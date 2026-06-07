"""Base class for single-path filesystem watchers using watchdog.

Provides debounced change detection and event dispatch so concrete watchers
only need to specify *what* to watch and override ``create_event()`` to
build a domain-specific event from the accumulated set of changed file paths.

Placed outside ``modules/`` so the service auto-discovery (which scans
``modules/`` and ``services/``) does not try to instantiate this class.
"""

from __future__ import annotations

import threading
from logging import Logger
from pathlib import Path
from typing import Callable, override

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers.api import BaseObserver

from .events import Event, ShutdownEvent, StartupEvent
from .modules.event_dispatcher import EventDispatcher, event_handler
from .modules.logging_factory import LoggingFactory
from .modules.service import Service
from .modules.thread_factory import ThreadFactory


class _FilterEventHandler(FileSystemEventHandler):
    """Watchdog handler that fires *on_change* with the changed file path."""

    def __init__(self, file_filter: Callable[[str], bool], on_change: Callable[[str], None]):
        self._file_filter: Callable[[str], bool] = file_filter
        self._on_change: Callable[[str], None] = on_change

    @override
    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._file_filter(str(event.src_path)):
            self._on_change(str(event.src_path))

    @override
    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._file_filter(str(event.src_path)):
            self._on_change(str(event.src_path))

    @override
    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._file_filter(str(event.src_path)):
            self._on_change(str(event.src_path))

    @override
    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            if self._file_filter(str(event.src_path)):
                self._on_change(str(event.src_path))
            elif self._file_filter(str(event.dest_path)):
                self._on_change(str(event.dest_path))


class SinglePathWatcherService(Service):
    """Watches a single filesystem path and emits an event on changes.

    Subclasses must override ``__init__`` to pass the keyword arguments
    (``watch_path``, ``file_filter``) to ``super()`` and override
    ``create_event()`` to build the domain-specific event from the
    accumulated set of changed file paths.
    """

    def __init__(
        self,
        event_dispatcher: EventDispatcher,
        logging: LoggingFactory,
        *,
        watch_path: str | Path,
        file_filter: Callable[[str], bool],
        debounce_seconds: float = 1.0,
        watch_label: str = "",
        observer: BaseObserver,
        thread_factory: ThreadFactory,
    ) -> None:
        self._logger: Logger = logging.get_logger(__name__)
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self._debounce_seconds: float = debounce_seconds
        self._watch_path: str = str(watch_path)
        self._file_filter: Callable[[str], bool] = file_filter
        self._watch_label: str = watch_label or type(self).__name__

        self._observer: BaseObserver = observer
        self._timer: threading.Timer | None = None
        self._pending_files: set[str] = set()
        self._lock: threading.Lock = threading.Lock()

        self._init_observer_thread: threading.Thread = thread_factory.create(target=self._start_observer, name=self._watch_label)

    # ------------------------------------------------------------------
    # To be overridden by subclasses
    # ------------------------------------------------------------------

    def create_event(self, changed_files: frozenset[str]) -> Event:
        """Build a domain-specific event from the accumulated changed file paths.

        Called from the debounce timer thread after the debounce window closes.
        *changed_files* contains the absolute paths of all files that triggered
        the watcher during the debounce window.
        """
        _ = changed_files
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @event_handler(StartupEvent)
    def on_startup(self, event: StartupEvent) -> None:  # pyright: ignore[reportUnusedParameter]
        self._init_observer_thread.start()

    @event_handler(ShutdownEvent)
    def on_shutdown(self, event: ShutdownEvent):  # pyright: ignore[reportUnusedParameter]
        self._observer.stop()

    def _start_observer(self) -> None:
        handler = _FilterEventHandler(self._file_filter, self._on_change)
        path = Path(self._watch_path)
        if path.is_dir():
            self._observer.schedule(handler, self._watch_path, recursive=False)
            self._logger.info(
                "%s watcher started. [path=%s, debounce=%.1fs]",
                self._watch_label,
                self._watch_path,
                self._debounce_seconds,
            )
        else:
            self._logger.warning(
                "%s watch path does not exist, skipping. [path=%s]",
                self._watch_label,
                self._watch_path,
            )
        self._observer.start()

    # ------------------------------------------------------------------
    # Debounce
    # ------------------------------------------------------------------

    def _on_change(self, file_path: str) -> None:
        """Debounced callback — called by ``_FilterEventHandler``."""
        with self._lock:
            self._pending_files.add(file_path)

            if self._timer is not None:
                self._timer.cancel()

            self._timer = threading.Timer(self._debounce_seconds, self._dispatch_change)
            self._timer.daemon = True

        self._timer.start()

    def _dispatch_change(self) -> None:
        """Dispatch the event (called from debounce timer thread)."""
        with self._lock:
            self._timer = None
            changed = frozenset(self._pending_files)
            self._pending_files.clear()

        self._logger.debug("Dispatching %s event [%d file(s)]", self._watch_label, len(changed))
        self._event_dispatcher.dispatch(self.create_event(changed))
