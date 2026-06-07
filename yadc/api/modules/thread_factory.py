"""Factory for creating and tracking daemon background threads.

Use in service constructors or lifecycle hooks that need a long-running
background thread:

    self._thread = thread_factory.create(target=self._run, name="my-watcher")
    self._thread.start()

The factory tracks every created thread so callers can join them at
shutdown. Threads default to ``daemon=True`` so the process can exit even
if the thread is still running — callers that need graceful shutdown are
responsible for unblocking the worker first (e.g. by signalling a stop
event) and then calling :meth:`join_all`.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from yadc.api.configuration import Configuration
from yadc.api.events import ShutdownEvent
from yadc.api.modules.event_dispatcher import event_handler

from .service import Service


class ThreadFactory(Service):
    """Singleton factory for daemon background threads."""

    def __init__(self, configuration: Configuration) -> None:
        self._threads: list[threading.Thread] = []
        self._lock: threading.Lock = threading.Lock()
        self._timeout: float = configuration.graceful_shutdown_threads_timeout
        self._event: threading.Event = threading.Event()

    @property
    def shutdown_event(self):
        """Event will be set during shutdown so threads running a background loop can stop"""
        return self._event

    def create(
        self,
        target: Callable[..., Any],
        *,
        name: str = "",
        daemon: bool = True,
    ) -> threading.Thread:
        """Build a Thread that runs *target*; caller is responsible for starting it.

        *daemon* defaults to ``True`` so the process can exit even if the
        thread is still running. Pass ``False`` to make the thread block
        process exit (rarely useful — prefer having the thread respond to
        a stop signal so :meth:`join_all` can wait for it cleanly).
        """
        thread = threading.Thread(target=target, name=name, daemon=daemon)
        with self._lock:
            self._threads.append(thread)

        return thread

    @event_handler(ShutdownEvent)
    def on_shutdown(self, event: ShutdownEvent):  # pyright: ignore[reportUnusedParameter]
        """Join all tracked threads with a *timeout* in seconds (``None`` = forever).

        Callers are responsible for unblocking the threads first (e.g. by
        signalling a stop event that the worker loop is polling); this
        method only waits for them to exit.
        """
        with self._lock:
            threads = list(self._threads)
            self._event.set()

        for thread in threads:
            thread.join(timeout=self._timeout)
