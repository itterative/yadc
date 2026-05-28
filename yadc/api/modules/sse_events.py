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
