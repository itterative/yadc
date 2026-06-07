"""Integration tests for SSE connection lifecycle and EventDispatcher async bridging."""

import asyncio
from threading import Event as ThreadEvent
from threading import Thread
from unittest.mock import MagicMock

import pytest

from yadc.api.configuration import Configuration
from yadc.api.events import CaptioningStatusEvent, SetupAppEvent, ShutdownEvent
from yadc.api.modules.event_dispatcher import EventDispatcher, event_handler
from yadc.api.modules.job_scheduler import JobScheduler
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.modules.sse_events import SSEEvents

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sse_events() -> tuple[SSEEvents, EventDispatcher]:
    """Wire up SSEEvents with real dependencies (no mocks for the event path)."""
    config = Configuration()
    logging_factory = LoggingFactory(config)
    event_dispatcher = EventDispatcher(logging_factory)
    job_scheduler = JobScheduler(logging_factory)
    sse = SSEEvents(config, event_dispatcher, job_scheduler, logging_factory)
    event_dispatcher.register_service(sse)
    return sse, event_dispatcher


class TestSSEReceiveCleanup:
    """SSE receive() generator cleanup paths — verify the per-client queue is
    always removed, whether the client disconnects, the server shuts down, or
    events flow normally."""

    @pytest.mark.asyncio
    async def test_queue_removed_on_client_cancel(self):
        """When the receive() generator is cancelled (client disconnect), the per-client queue must be removed."""
        sse, ed = _make_sse_events()

        # Capture the event loop so the dispatcher can bridge async handlers.
        ed.set_loop(asyncio.get_running_loop())

        assert len(sse._queues) == 0

        gen = sse.receive(object)

        # Start iterating the generator (registers the queue, then blocks waiting for events).
        anext_task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0.05)

        # The generator should have registered a queue.
        assert len(sse._queues) == 1

        # Simulate client disconnect by closing the generator.
        anext_task.cancel()
        try:
            await anext_task
        except (asyncio.CancelledError, StopAsyncIteration):
            pass
        await gen.aclose()

        # Queue must be removed by the finally block.
        assert len(sse._queues) == 0

    @pytest.mark.asyncio
    async def test_queue_removed_on_shutdown(self):
        """When ShutdownEvent is dispatched, the receive() generator should exit and clean up its queue."""
        sse, ed = _make_sse_events()

        assert len(sse._queues) == 0

        # Start a background task that consumes from the generator.
        received_items: list[tuple[int, object]] = []

        async def _consume():
            async for item in sse.receive(object):
                received_items.append(item)

        task = asyncio.create_task(_consume())

        # Give the event loop a tick so the generator registers its queue.
        await asyncio.sleep(0.05)
        assert len(sse._queues) == 1

        # Dispatch shutdown — this sets _shutdown=True, which causes receive() to exit.
        ed.dispatch(ShutdownEvent())

        # Wait for the consumer task to finish.
        await asyncio.wait_for(task, timeout=2.0)

        # Queue must have been removed by the finally block.
        assert len(sse._queues) == 0

    @pytest.mark.asyncio
    async def test_events_are_delivered(self):
        """Events dispatched through EventDispatcher reach the SSE receive() generator."""
        sse, ed = _make_sse_events()

        received: list[tuple[int, object]] = []

        async def _consume():
            async for item in sse.receive(CaptioningStatusEvent):
                received.append(item)
                if len(received) >= 2:
                    break

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0.05)

        # Push two events.
        ed.dispatch(CaptioningStatusEvent(status="running", dataset_name="ds1", processed=1, total=10, errors=0))
        ed.dispatch(CaptioningStatusEvent(status="running", dataset_name="ds1", processed=2, total=10, errors=0))

        await asyncio.wait_for(task, timeout=2.0)

        assert len(received) == 2
        # event IDs should be monotonically increasing.
        assert received[0][0] < received[1][0]
        assert received[0][1].dataset_name == "ds1"
        assert received[1][1].processed == 2


class TestEventDispatcherBridge:
    """EventDispatcher thread-to-async bridge — verify that handlers registered
    via ``@event_handler`` are invoked correctly when ``dispatch()`` is called
    from a background thread, for both async and sync handlers."""

    @pytest.mark.asyncio
    async def test_async_handler_runs_on_main_loop(self):
        """An async handler registered via @event_handler is correctly invoked when
        dispatch() is called from a background thread (run_coroutine_threadsafe path)."""

        # Use a fresh event loop so we control exactly when it runs.
        loop = asyncio.get_running_loop()

        config = Configuration()
        logging_factory = LoggingFactory(config)
        ed = EventDispatcher(logging_factory)
        ed.set_loop(loop)

        # Track whether the async handler ran and on which loop.
        handler_called = ThreadEvent()
        handler_loop_id: int | None = [None]  # mutable container for closure

        class TestEvent(CaptioningStatusEvent):
            TYPE = "test_bridge"  # pyright: ignore[reportIncompatibleVariableOverride]

        class FakeService:
            @event_handler(TestEvent)
            async def on_test(self, event: TestEvent) -> None:
                handler_loop_id[0] = id(asyncio.get_running_loop())
                handler_called.set()

        svc = FakeService()
        ed.register_service(svc)

        # Dispatch from a background thread.
        thread = Thread(
            target=ed.dispatch,
            args=(TestEvent(status="running", dataset_name="test", processed=0, total=1, errors=0),),
        )
        thread.start()

        # Run the loop briefly to process the scheduled coroutine.
        # We need to yield control so the loop can execute the task.
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, handler_called.wait, 3),
                timeout=5,
            )
        except TimeoutError:
            pytest.fail("Async handler was not called within timeout")

        thread.join(timeout=2)

        assert handler_called.is_set(), "Async handler was never called"
        assert handler_loop_id[0] == id(loop), "Handler ran on a different event loop than expected"

    @pytest.mark.asyncio
    async def test_sync_handler_invoked_directly(self):
        """A sync handler works normally when dispatch() is called from a background thread."""

        config = Configuration()
        logging_factory = LoggingFactory(config)
        ed = EventDispatcher(logging_factory)

        handler_called = ThreadEvent()
        received_status: list[str] = []

        class TestSyncEvent(CaptioningStatusEvent):
            TYPE = "test_sync_bridge"  # pyright: ignore[reportIncompatibleVariableOverride]

        class FakeSyncService:
            @event_handler(TestSyncEvent)
            def on_test(self, event: TestSyncEvent) -> None:
                received_status.append(event.status)
                handler_called.set()

        svc = FakeSyncService()
        ed.register_service(svc)

        thread = Thread(
            target=ed.dispatch,
            args=(TestSyncEvent(status="done", dataset_name="test", processed=5, total=5, errors=0),),
        )
        thread.start()
        thread.join(timeout=2)

        assert handler_called.is_set()
        assert received_status == ["done"]


class TestSetupAppEvent:
    """``SetupAppEvent`` carries the Quart app so services can attach request/
    response hooks (e.g. CORS) before blueprints are registered. Verify
    handlers receive the app passed to ``dispatch()``."""

    def test_handler_receives_app_from_event(self):
        config = Configuration()
        logging_factory = LoggingFactory(config)
        ed = EventDispatcher(logging_factory)

        received_app: list[object] = []

        class FakeService:
            @event_handler(SetupAppEvent)
            def on_setup_app(self, event: SetupAppEvent) -> None:
                received_app.append(event.app)

        svc = FakeService()
        ed.register_service(svc)

        mock_app = MagicMock()
        ed.dispatch(SetupAppEvent(app=mock_app))

        assert received_app == [mock_app], "SetupAppEvent handler did not receive the Quart app"
