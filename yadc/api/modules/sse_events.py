import itertools
from collections import deque
from collections.abc import Generator
from logging import Logger
from threading import Condition
from typing import Any

from ..configuration import Configuration
from ..events import CaptioningStatusEvent, DatasetChangedEvent, Event, PingEvent, ResumptionFailedEvent, ShutdownEvent
from .event_dispatcher import EventDispatcher, event_handler
from .job_scheduler import JobScheduler
from .logging_factory import LoggingFactory
from .service import Service

SSE_EVENT_PING = "ping"

# Per-client SSE queue items: (event_id, event).
# Ping events use id 0; real events use monotonically increasing IDs.
type _QueueItem = tuple[int, Event]


class SSEEvents(Service):
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
        self._queues: list[list[_QueueItem]] = []
        self._shutdown: bool = False

        # Event ID counter and history ring buffer for Last-Event-ID resumption.
        # Ping events are excluded from history.
        self._event_counter: itertools.count[int] = itertools.count(1)
        self._history: deque[tuple[int, Event]] = deque(maxlen=configuration.sse_event_history_size)

        self._did_warn: bool = False
        self._did_error: bool = False

        job_scheduler.new_scheduled_job(5, self._send_ping)

    def _send_ping(self):
        from datetime import datetime, timezone

        self.push(PingEvent(time=datetime.now(timezone.utc).isoformat()))

    @event_handler(ShutdownEvent)
    def on_shutdown(self, _event: ShutdownEvent):
        self._logger.info("Shutdown event received, stopping %d listeners", len(self._queues))
        with self._queue_cv:
            self._shutdown = True
            self._queue_cv.notify_all()

    @event_handler(CaptioningStatusEvent)
    def on_captioning_status(self, event: CaptioningStatusEvent):
        self.push(event)

    @event_handler(DatasetChangedEvent)
    def on_dataset_changed(self, event: DatasetChangedEvent):
        self.push(event)

    def push(self, event: Event):
        with self._queue_cv:
            is_ping = isinstance(event, PingEvent)

            if not is_ping:
                self._logger.debug("New sse event. [type=%s, listeners=%d]", event.TYPE, len(self._queues))

                event_id = next(self._event_counter)
                self._history.append((event_id, event))
            else:
                event_id = 0

            for queue in self._queues:
                if len(queue) > self.configuration.sse_listener_max_events:
                    continue

                queue.append((event_id, event))

                if len(queue) >= self.configuration.sse_listener_max_events:
                    self._logger.warning(
                        "An SSE event listener has reached the maximum number of events (%d) in the queue.",
                        self.configuration.sse_listener_max_events,
                    )

            self._queue_cv.notify_all()

    def receive(
        self,
        event_cls: type,
        last_event_id: int | None = None,
    ) -> Generator[tuple[int, Event], Any, None]:
        """Yield ``(event_id, event)`` tuples filtered by *event_cls*.

        If *last_event_id* is provided (from the SSE ``Last-Event-ID`` header),
        missed events are replayed from the in-memory history first.  When the
        requested ID is older than the oldest entry in the ring buffer, a single
        ``ResumptionFailedEvent`` is yielded before the live stream begins so
        the client can warn the user.
        """
        with self._queue_cv:
            if len(self._queues) >= self.configuration.sse_listeners_max:
                if not self._did_error:
                    self._logger.warning("Number of SSE listeners have reached %d. No longer accepting listeners.", len(self._queues))
                    self._did_error = True

                return

            # Register the client queue **before** snapshotting history so that
            # no events pushed between the snapshot and queue registration are
            # lost.  Because both happen under _queue_cv, there is no race.
            queue: list[_QueueItem] = []
            self._queues.append(queue)
            self._logger.debug("New sse events listener. [listeners=%d]", len(self._queues))

            if len(self._queues) >= self.configuration.sse_listeners_warning:
                if not self._did_warn:
                    self._logger.warning("Number of SSE listeners have reached %d.", len(self._queues))
                    self._did_warn = True
            else:
                self._did_warn = False
                self._did_error = False

            # Snapshot replay events while still holding the lock.
            replay_events: list[_QueueItem] = []
            resumption_failed: bool = False
            failed_id: int = 0

            if last_event_id is not None:
                if self._history and last_event_id < self._history[0][0]:
                    # Requested ID is older than our oldest buffered event.
                    resumption_failed = True
                    failed_id = last_event_id
                    self._logger.info(
                        "SSE resumption failed: requested id %d is older than oldest buffered id %d",
                        last_event_id,
                        self._history[0][0],
                    )
                else:
                    replay_events = [(eid, ev) for eid, ev in self._history if eid > last_event_id]
                    if replay_events:
                        self._logger.debug("SSE resumption: replaying %d events after id %d", len(replay_events), last_event_id)

        try:
            # Phase 1: replay missed history (or signal failure).
            if resumption_failed:
                yield (0, ResumptionFailedEvent(requested_event_id=failed_id))

            for eid, ev in replay_events:
                if isinstance(ev, event_cls):
                    yield (eid, ev)

            # Phase 2: live stream.
            while not self._shutdown:
                with self._queue_cv:
                    self._queue_cv.wait(timeout=1)

                    if self._shutdown:
                        break

                    items: list[_QueueItem] = []
                    while queue:
                        items.append(queue.pop(0))

                yield from ((eid, ev) for eid, ev in items if isinstance(ev, event_cls))
        finally:
            with self._queue_cv:
                try:
                    self._queues.remove(queue)
                    self._logger.debug("Removed sse events listener. [listeners=%d]", len(self._queues))
                except ValueError:
                    pass
