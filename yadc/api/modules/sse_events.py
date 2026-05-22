from collections.abc import Generator
from logging import Logger
from threading import Condition
from typing import Any

from injector import inject, singleton

from ..configuration import Configuration
from ..events import CaptioningStatusEvent, Event, PingEvent
from .event_dispatcher import EventDispatcher, event_handler
from .job_scheduler import JobScheduler
from .logging_factory import LoggingFactory
from .service import Service

SSE_EVENT_PING = "ping"


@singleton
class SSEEvents(Service):
    @inject
    def __init__(
        self,
        configuration: Configuration,
        event_dispatcher: EventDispatcher,
        job_scheduler: JobScheduler,
        logging: LoggingFactory,
    ):
        self._logger: Logger = logging.get_logger(__name__)

        self.configuration: Configuration = configuration

        self._queue_cv: Condition = Condition()
        self._queues: list[list[Event]] = []

        self._did_warn: bool = False
        self._did_error: bool = False

        job_scheduler.new_scheduled_job(5, self._send_ping)

        event_dispatcher.register_service(self)

    def _send_ping(self):
        from datetime import datetime, timezone

        self.push(PingEvent(time=datetime.now(timezone.utc).isoformat()))

    @event_handler(CaptioningStatusEvent)
    def on_captioning_status(self, event: CaptioningStatusEvent):
        self.push(event)

    def push(self, event: Event):
        with self._queue_cv:
            if not isinstance(event, PingEvent):
                self._logger.debug("New sse event. [type=%s, listeners=%d]", event.TYPE, len(self._queues))

            for queue in self._queues:
                if len(queue) > self.configuration.sse_listener_max_events:
                    continue

                queue.append(event)

                if len(queue) >= self.configuration.sse_listener_max_events:
                    self._logger.warning(
                        "An SSE event listener has reached the maximum number of events (%d) in the queue.",
                        self.configuration.sse_listener_max_events,
                    )

            self._queue_cv.notify_all()

    def receive(self, event_cls: type) -> Generator[Event, Any, None]:
        with self._queue_cv:
            if len(self._queues) >= self.configuration.sse_listeners_max:
                if not self._did_error:
                    self._logger.warning("Number of SSE listeners have reached %d. No longer accepting listeners.", len(self._queues))
                    self._did_error = True

                return

            queue: list[Event] = []
            self._queues.append(queue)
            self._logger.debug("New sse events listener. [listeners=%d]", len(self._queues))

            if len(self._queues) >= self.configuration.sse_listeners_warning:
                if not self._did_warn:
                    self._logger.warning("Number of SSE listeners have reached %d.", len(self._queues))
                    self._did_warn = True
            else:
                self._did_warn = False
                self._did_error = False

        try:
            while True:
                with self._queue_cv:
                    self._queue_cv.wait(timeout=5)

                    events: list[Event] = []
                    while queue:
                        events.append(queue.pop(0))

                yield from (e for e in events if isinstance(e, event_cls))
        finally:
            with self._queue_cv:
                try:
                    self._queues.remove(queue)
                    self._logger.debug("Removed sse events listener. [listeners=%d]", len(self._queues))
                except ValueError:
                    pass
