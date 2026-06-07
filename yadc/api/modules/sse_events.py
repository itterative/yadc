"""``SSEEvents`` — SSE fan-out with monotonic event IDs and ``Last-Event-ID`` replay.

Subscribes to all of the API's event types via ``@event_handler`` (see
``event_dispatcher``) and fans them out to one ``asyncio.Queue`` per
SSE client. Each real event is assigned a monotonically increasing id
(``itertools.count(1)``) and pushed onto a ring buffer
(``Configuration.sse_event_history_size``); ``PingEvent`` uses id 0 and
is not buffered.

``receive(event_cls, last_event_id)`` is the per-client async generator
called by ``api_events``:

- If ``last_event_id`` is set, missed events are replayed from the
  buffer (filtered to those > ``last_event_id`` and of the requested
  class) *before* the live stream begins.
- If ``last_event_id`` is older than the oldest buffered id, a single
  ``ResumptionFailedEvent`` is yielded so the client can warn the user.
- On disconnect (Quart cancels the generator), the per-client queue is
  removed in the ``finally`` block and the listener count drops back.

Listens to ``ShutdownEvent`` to stop accepting new events so clients
get a clean close. ``JobScheduler`` sends a ``PingEvent`` every 5s.
Listener-count thresholds (``sse_listeners_warning`` /
``sse_listeners_max``) emit warnings or refuse new listeners.
"""

import asyncio
import itertools
from collections import deque
from collections.abc import AsyncGenerator
from logging import Logger

from ..configuration import Configuration
from ..events import (
    CaptioningStatusEvent,
    DatasetChangedEvent,
    EnvironmentsChangedEvent,
    Event,
    ImageCaptionedEvent,
    ImageCaptionErrorEvent,
    ImageCaptionStartedEvent,
    ImageRefinedEvent,
    PingEvent,
    ResumptionFailedEvent,
    ShutdownEvent,
    TemplatesChangedEvent,
)
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
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self.configuration: Configuration = configuration

        self._lock: asyncio.Lock = asyncio.Lock()
        self._queues: list[asyncio.Queue[_QueueItem]] = []
        self._shutdown: bool = False

        # Event ID counter and history ring buffer for Last-Event-ID resumption.
        # Ping events are excluded from history.
        self._event_counter: itertools.count[int] = itertools.count(1)
        self._history: deque[tuple[int, Event]] = deque(maxlen=configuration.sse_event_history_size)

        self._did_warn: bool = False
        self._did_error: bool = False

        job_scheduler.new_scheduled_job(5, self._send_ping)

    def _send_ping(self) -> None:
        from datetime import datetime, timezone

        self._event_dispatcher.dispatch(PingEvent(time=datetime.now(timezone.utc).isoformat()))

    @event_handler(ShutdownEvent)
    def on_shutdown(self, _event: ShutdownEvent) -> None:
        self._logger.info("Shutdown event received, stopping %d listeners", len(self._queues))
        self._shutdown = True

    @event_handler(PingEvent)
    async def on_ping(self, event: PingEvent) -> None:
        await self.push(event)

    @event_handler(CaptioningStatusEvent)
    async def on_captioning_status(self, event: CaptioningStatusEvent) -> None:
        await self.push(event)

    @event_handler(DatasetChangedEvent)
    async def on_dataset_changed(self, event: DatasetChangedEvent) -> None:
        await self.push(event)

    @event_handler(EnvironmentsChangedEvent)
    async def on_environments_changed(self, event: EnvironmentsChangedEvent) -> None:
        await self.push(event)

    @event_handler(TemplatesChangedEvent)
    async def on_templates_changed(self, event: TemplatesChangedEvent) -> None:
        await self.push(event)

    @event_handler(ImageCaptionedEvent)
    async def on_image_captioned(self, event: ImageCaptionedEvent) -> None:
        await self.push(event)

    @event_handler(ImageCaptionStartedEvent)
    async def on_image_caption_started(self, event: ImageCaptionStartedEvent) -> None:
        await self.push(event)

    @event_handler(ImageRefinedEvent)
    async def on_image_refined(self, event: ImageRefinedEvent) -> None:
        await self.push(event)

    @event_handler(ImageCaptionErrorEvent)
    async def on_image_caption_error(self, event: ImageCaptionErrorEvent) -> None:
        await self.push(event)

    async def push(self, event: Event) -> None:
        async with self._lock:
            is_ping = isinstance(event, PingEvent)

            if not is_ping:
                self._logger.debug("New sse event. [type=%s, listeners=%d]", event.TYPE, len(self._queues))
                event_id = next(self._event_counter)
                self._history.append((event_id, event))
            else:
                event_id = 0

            for queue in self._queues:
                if queue.full():
                    if not self._did_warn:
                        self._logger.warning(
                            "An SSE event listener has reached the maximum number of events (%d) in the queue.",
                            self.configuration.sse_listener_max_events,
                        )
                        self._did_warn = True
                    continue

                queue.put_nowait((event_id, event))

            self._did_warn = False

    async def receive(
        self,
        event_cls: type,
        last_event_id: int | None = None,
    ) -> AsyncGenerator[tuple[int, Event], None]:
        """Yield ``(event_id, event)`` tuples filtered by *event_cls*.

        If *last_event_id* is provided (from the SSE ``Last-Event-ID`` header),
        missed events are replayed from the in-memory history first.  When the
        requested ID is older than the oldest entry in the ring buffer, a single
        ``ResumptionFailedEvent`` is yielded before the live stream begins so
        the client can warn the user.

        The async generator is cancelled by Quart when the client disconnects;
        the ``finally`` block ensures the per-client queue is removed.
        """
        async with self._lock:
            if len(self._queues) >= self.configuration.sse_listeners_max:
                if not self._did_error:
                    self._logger.warning(
                        "Number of SSE listeners have reached %d. No longer accepting listeners.",
                        len(self._queues),
                    )
                    self._did_error = True
                return

            queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=self.configuration.sse_listener_max_events)
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
            if resumption_failed:
                yield (0, ResumptionFailedEvent(requested_event_id=failed_id))

            for eid, ev in replay_events:
                if isinstance(ev, event_cls):
                    yield (eid, ev)

            while not self._shutdown:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                eid, ev = item
                if isinstance(ev, event_cls):
                    yield (eid, ev)
        finally:
            async with self._lock:
                try:
                    self._queues.remove(queue)
                    self._logger.debug("Removed sse events listener. [listeners=%d]", len(self._queues))
                except ValueError:
                    pass
