---
date: 2026-05-27
---

# Phase 2: Async SSE and EventDispatcher Bridge

**Context:** Phase 1 migrated the server to Quart/uvicorn but left SSE as a sync generator wrapped in `asyncio.to_thread()`. This worked but was a temporary hack. The proper fix was making SSE fully async so it doesn't block the event loop or consume thread-pool workers.

**Decision:** Rewrite `SSEEvents` with `asyncio.Queue` per client, add an async bridge in `EventDispatcher`, and make the `/events` endpoint a native async generator.

**Rationale:**
- `threading.Condition` is not asyncio-safe and blocks the event loop.
- Background threads (`CaptionJob`) need to push events safely into the async world.
- Quart cancels async generators on client disconnect, giving us clean teardown without polling.

**What was done:**
- `yadc/api/modules/sse_events.py` (full rewrite):
  - Replaced `threading.Condition` with `asyncio.Lock`.
  - Replaced `list[list[_QueueItem]]` with `list[asyncio.Queue[_QueueItem]]`.
  - Made `on_ping`, `on_captioning_status`, `on_dataset_changed` async handlers.
  - Made `push()` async (acquires lock, assigns ID, `queue.put_nowait`).
  - Made `receive()` async (registers queue, replays history, `await asyncio.wait_for(queue.get(), timeout=1.0)`).
  - Client cleanup in `try/finally`.
  - `_send_ping()` now dispatches via `EventDispatcher` instead of calling `push()` directly (so the bridge handles it).
- `yadc/api/modules/event_dispatcher.py`:
  - Added `_loop: asyncio.AbstractEventLoop | None`.
  - Added `set_loop()` method.
  - `dispatch()` now inspects handlers with `inspect.iscoroutinefunction()`.
  - Async handlers are scheduled via `asyncio.create_task` (same thread) or `asyncio.run_coroutine_threadsafe` (background thread).
- `yadc/api/application.py`:
  - Added `@app.before_serving` hook that calls `event_dispatcher.set_loop(asyncio.get_running_loop())`.
  - Fixed docstring (`Flask` → `Quart`).
- `yadc/api/controllers/api_events.py`:
  - Removed `asyncio.to_thread()` wrapper (no longer needed since `receive()` is async).
  - Native `async for event_id, event in sse_events.receive(...)`.
  - Added `try/except asyncio.CancelledError` around the generator body to suppress uvicorn's `ASGI callable returned without completing response` error on client disconnect.

**Verification:**
- `ruff check yadc tests` — clean.
- `basedpyright yadc/api` — 0 errors in changed files.
- `pytest tests/` — 128 passed, 14 skipped.
- SSE endpoint serves pings, captioning status, and dataset_changed events correctly.
- Client disconnect produces no ASGI errors in uvicorn logs.
- Captioning flow works end-to-end through the webui.

**Files touched:**
- `yadc/api/modules/sse_events.py`
- `yadc/api/modules/event_dispatcher.py`
- `yadc/api/application.py`
- `yadc/api/controllers/api_events.py`
