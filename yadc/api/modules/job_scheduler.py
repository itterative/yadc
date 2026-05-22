from logging import Logger
from threading import Lock, Thread
from typing import Callable

from .logging_factory import LoggingFactory
from .service import Service


class JobScheduler(Service):
    def __init__(self, logging: LoggingFactory):
        self._logger: Logger = logging.get_logger(__name__)
        self._jobs: list[Thread] = []
        self._lock: Lock = Lock()

        self._new_job(self._cleanup)

    def new_scheduled_job[T, **P](self, interval: float, target: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        self._new_scheduled_job_with_delay(interval, 0, target, *args, **kwargs)

    def _new_scheduled_job_with_delay[T, **P](self, interval: float, delay: float, target: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        def _wrapper():
            import time

            time.sleep(delay)

            while True:
                try:
                    target(*args, **kwargs)
                except Exception as e:
                    self._logger.exception("Unhandled exception in background thread %s: %s", target.__qualname__, e)

                time.sleep(interval)

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

        while True:
            time.sleep(10)

            with self._lock:
                self._jobs = list(filter(lambda j: j.is_alive(), self._jobs))
