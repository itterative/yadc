"""``JobScheduler`` — daemon threads for running periodic background jobs."""

from logging import Logger
from threading import Lock, Thread
from typing import Callable

from ..configuration import Configuration
from ..events import ShutdownEvent
from .event_dispatcher import event_handler
from .logging_factory import LoggingFactory
from .service import Service

_CLEANUP_INTERVAL = 10


class JobScheduler(Service):
    def __init__(self, logging: LoggingFactory, configuration: Configuration):
        self._logger: Logger = logging.get_logger(__name__)
        self._jobs: list[Thread] = []
        self._lock: Lock = Lock()
        # Checked between ticks so scheduled threads stop promptly on
        # shutdown instead of firing into a dead event loop.
        self._shutdown: bool = False
        # Matches ``ThreadFactory``; daemon threads are killed on process exit if they overrun it.
        self._timeout: float = configuration.graceful_shutdown_threads_timeout

        self._new_job(self._cleanup)

    @event_handler(ShutdownEvent)
    def on_shutdown(self, _event: ShutdownEvent) -> None:
        with self._lock:
            self._shutdown = True
            count = len(self._jobs)
        self._logger.info("Stopping %d background job threads. [timeout=%.1fs]", count, self._timeout)
        for thread in self._jobs:
            try:
                # Threads check ``_shutdown`` between ticks; the join is a safety net for runaway targets.
                thread.join(timeout=self._timeout)
            except RuntimeError:
                # ``Thread.join`` raises if the thread was never started.
                pass
        with self._lock:
            alive = [t for t in self._jobs if t.is_alive()]
        if alive:
            self._logger.warning(
                "Background jobs did not stop within timeout. [alive=%s]",
                [t.name for t in alive],
            )

    def new_scheduled_job[T, **P](self, interval: float, target: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        self._new_scheduled_job_with_delay(interval, 0, target, *args, **kwargs)

    def _new_scheduled_job_with_delay[T, **P](self, interval: float, delay: float, target: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        def _wrapper():
            import time

            time.sleep(delay)

            # Sleep in short chunks so ``on_shutdown`` is responsive
            # even when ``interval`` is large (the dataset refresh
            # interval can be hours).
            chunk = 0.1

            while not self._shutdown:
                try:
                    target(*args, **kwargs)
                except Exception as e:
                    self._logger.exception("Unhandled exception in background thread %s: %s", target.__qualname__, e)

                slept = 0.0
                while slept < interval and not self._shutdown:
                    time.sleep(min(chunk, interval - slept))
                    slept += chunk

        with self._lock:
            thread = Thread(target=_wrapper, daemon=True)
            thread.start()
            self._jobs.append(thread)

    def _new_job[T, **P](self, target: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        with self._lock:
            thread = Thread(target=target, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            self._jobs.append(thread)

    def _cleanup(self):
        import time

        # Chunked sleep so ``on_shutdown`` joining this thread sees it exit
        # within ``_timeout`` instead of always landing in the "did not stop"
        # warning. 10s cadence unchanged.
        chunk = 0.1
        while not self._shutdown:
            slept = 0.0
            while slept < _CLEANUP_INTERVAL and not self._shutdown:
                time.sleep(min(chunk, _CLEANUP_INTERVAL - slept))
                slept += chunk

            with self._lock:
                self._jobs = list(filter(lambda j: j.is_alive(), self._jobs))
