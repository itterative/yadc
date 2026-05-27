---
name: async-migration-plan
description: Phased plan for migrating the API backend from Flask/waitress to Quart/uvicorn, making SSE fully async, then deferring a potential FastAPI migration.
last_history: 3
---

# Async API Migration Plan (Quart)

## Context

The current API backend is built on **Flask + waitress** (WSGI). SSE connections block a worker thread indefinitely, causing:
- Ctrl+C hangs (waitress threads don't unblock).
- TCP proxy zombie connections (no way to detect dead clients without writing).
- Thread pool exhaustion (one thread per SSE listener).

**Flask 2.x supports `async def` views, but only within WSGI** — it spins up a temporary event loop per request inside the worker thread. The thread is still occupied for the entire request, and `Response` only accepts synchronous iterables, so SSE generators cannot be async.

The solution is **Quart + uvicorn** (ASGI). Quart is API-compatible with Flask (same Blueprints, `request`, `send_file`, `jsonify`, `after_request`), but natively supports async generators as response bodies and runs on an asyncio event loop. This gives us non-blocking SSE with minimal controller churn.

A future **FastAPI** migration is still possible after this refactor, but it is deliberately out of scope. This plan is strictly about making the backend async.

## Goals

1. **Swap framework**: Flask → Quart, waitress → uvicorn.
2. **Async SSE**: `/events` becomes a non-blocking async generator; client disconnect cancels it natively.
3. **Thread-to-async bridge**: `EventDispatcher` schedules async handlers on the main loop so background threads (CaptionJob, JobScheduler) can safely push events.
4. **Keep injector + auto-discovery**: Zero changes to service constructors or the `@controller` pattern.
5. **Minimal controller changes**: Only routes using streaming or `send_file` need to become `async def`.
6. **Clean shutdown**: `SIGINT` cleanly cancels SSE generators and background jobs.
7. **No frontend changes**: Same REST + SSE interface.

## Architectural Decisions

### Quart, not FastAPI (for now)

Quart was chosen because:
- **API-compatible with Flask**: `Blueprint`, `request`, `send_file`, `jsonify`, `after_request`, `stream_with_context` all exist with the same signatures.
- **Controllers stay almost identical**: Only `api_events.py` and a few `send_file` routes need `async def`.
- **Low migration risk**: We can validate async SSE works end-to-end before committing to a larger framework rewrite.
- **FastAPI can follow later**: Once services are async-ready, migrating from Quart to FastAPI is a second, smaller refactor focused on request/response models.

### EventDispatcher: sync dispatch, async handler bridge

`EventDispatcher.dispatch()` is called from:
- The main asyncio thread (e.g. `DatasetWatcherService` callbacks).
- Background daemon threads (`CaptionJob._emit_status()`).

After the migration, some handlers (e.g. `SSEEvents.on_captioning_status`) are async coroutines. `dispatch()` stays **synchronous** (to avoid changing every caller), but:

1. `EventDispatcher` captures a reference to the main event loop during startup.
2. `dispatch()` inspects each handler with `inspect.iscoroutinefunction()`.
3. If the handler is async:
   - Called from the main loop thread → schedule it with `asyncio.create_task(handler(event))`.
   - Called from a background thread → schedule it with `asyncio.run_coroutine_threadsafe(handler(event), self._loop)`.
4. If the handler is sync → call it directly, same as today.

This is a localized change in `yadc/api/modules/event_dispatcher.py`. No service or controller code changes except adding `async def` to handler methods that need it.

### SSE: threading.Condition → asyncio.Queue

`SSEEvents` is the only module that needs a full rewrite:

- **Push (from any thread)**: `push(event)` is called by `on_captioning_status`, which is now async. It acquires an `asyncio.Lock`, assigns a monotonic ID, appends to the history ring buffer, and puts the item into every client's `asyncio.Queue`.
- **Per-client queues**: Each SSE listener gets an `asyncio.Queue` instead of a `list`.
- **Receive (async)**: `receive()` becomes `async def`. It registers a queue, replays history, then loops with `await queue.get()`.
- **Client disconnect**: Quart cancels the async generator when the client goes away; a `try/finally` block removes the queue.
- **Ping**: `JobScheduler` still runs `_send_ping` in a background thread. The thread calls `event_dispatcher.dispatch(PingEvent)`, which bridges to the async handler on the main loop.

### Controllers: what changes

| Controller | Changes |
|---|---|
| `api_events.py` | `async def` route; `_retrieve_events()` becomes `async def` using `async for`; remove `stream_with_context` and `waitress.client_disconnected`; return `quart.Response(async_gen, mimetype="text/event-stream")` |
| `api_datasets.py` | `serve_image_media` and `serve_image_thumbnail` become `async def` so they can `await quart.send_file(...)` |
| `app_frontend.py` | `index_endpoint`, `robots_endpoint`, `webmanifest_endpoint`, `static_root_file_endpoint`, `app_endpoint` become `async def` so they can `await quart.send_file(...)` / `await quart.send_from_directory(...)` |
| All other controllers (`api_captioning.py`, `api_configs.py`, `api_envs.py`, `api_export.py`, `api_templates.py`) | **No changes** — sync routes continue to work in Quart |

### Static / SPA Serving

Quart's `send_from_directory` and `send_file` are async. The `app_frontend.py` controller registers the same routes but as `async def` views. No structural change to the route table.

### CORS

`CORSMiddleware` uses `@blueprint.after_request`. Quart Blueprints support `after_request` identically to Flask. No code changes.

### Server Runner

`Application.run()`:

1. `configure_services()` and `configure_controllers()` (same as today).
2. `configure_app()` registers blueprints on the Quart app.
3. Dispatch `StartupEvent`.
4. Call `uvicorn.run(self.app, host=..., port=..., log_level=...)`.
5. Wrap in `try/finally` to dispatch `ShutdownEvent` when uvicorn returns (e.g. on SIGINT).

Remove the manual `signal.signal(SIGINT, ...)` handler. Uvicorn handles signal propagation; when it exits, the `finally` block dispatches `ShutdownEvent`, which unblocks SSE listeners.

## Phased Implementation

### Phase 1: Foundation — Quart, uvicorn, DI, CORS, Static Files

**Goal**: Server boots on Quart + uvicorn. All routes work. SSE still sync (will be fixed in Phase 2).

- `pyproject.toml`: swap `flask` → `quart`, `waitress` → `uvicorn[standard]`.
- `yadc/api/controllers/blueprints.py`: import `quart.Blueprint` instead of `flask.Blueprint`.
- `yadc/api/application.py`:
  - Replace `Flask` with `Quart`.
  - Bind `Quart` in the injector instead of `Flask`.
  - `configure_app()` registers blueprints (same API).
  - Replace `waitress.serve()` with `uvicorn.run(self.app, ...)`.
  - Remove the manual `SIGINT` handler; wrap `uvicorn.run()` in `try/finally` to dispatch `ShutdownEvent`.
- `yadc/api/modules/cors_middleware.py`: No changes (Quart Blueprints support `after_request`).
- `yadc/api/controllers/app_frontend.py`: Make all routes `async def`; replace `flask.send_file` / `send_from_directory` with `quart.send_file` / `send_from_directory` (awaited).
- `yadc/api/controllers/api_datasets.py`: Make `serve_image_media` and `serve_image_thumbnail` `async def`; await `quart.send_file`.
- `yadc/cli_webui.py`: Remove `--threads` option (uvicorn is async; threads are irrelevant). Keep other options.
- `yadc/api/configuration.py`: Remove `http_threads` field.

**Verify**: `uv run yadc webui serve` boots. Frontend loads. API calls succeed. SSE connects but still blocks a thread (Phase 2 will fix).

### Phase 2: Async SSE and EventDispatcher Bridge

**Goal**: SSE is fully async and non-blocking. Background threads can safely push events.

- `yadc/api/modules/event_dispatcher.py`:
  - Add `_loop: asyncio.AbstractEventLoop | None`.
  - In `dispatch()`, detect async handlers.
  - Schedule async handlers on the loop via `create_task` (same thread) or `run_coroutine_threadsafe` (background thread).
- `yadc/api/modules/sse_events.py`:
  - Replace `threading.Condition` with `asyncio.Lock`.
  - Replace `list[list[_QueueItem]]` with `list[asyncio.Queue[_QueueItem]]`.
  - Make `on_captioning_status`, `on_dataset_changed`, and `on_shutdown` `async def`.
  - Make `push()` `async def` (acquires lock, assigns ID, puts into queues).
  - Make `receive()` `async def` (registers queue, replays history, `await queue.get()` loop).
  - Client cleanup in `try/finally`.
- `yadc/api/controllers/api_events.py`:
  - Make `events_endpoint()` `async def`.
  - Make `_retrieve_events()` `async def` using `async for event_id, event in sse_events.receive(...)`.
  - Remove `stream_with_context` (not needed for async generators in Quart).
  - Remove `waitress.client_disconnected` check (Quart cancels the generator on disconnect).
  - Return `quart.Response(_retrieve_events(), mimetype="text/event-stream")`.

**Verify**:
- Connect to `/api/events`; pings arrive.
- Disconnect/reconnect works.
- `Last-Event-ID` resumption works.
- Start a captioning job; SSE progress events arrive.
- Ctrl+C with an active SSE tab shuts down cleanly within 2 seconds.

### Phase 3: Cleanup, Testing, and Validation

**Goal**: Remove Flask/waitress remnants, verify stability, fix shutdown bugs, document deferred FastAPI work.

- Remove all `flask` imports from `yadc/api/`.
- Remove all `waitress` references.
- Update any tests under `tests/` that import `flask.testing` → use `quart.testing` or `httpx`.
- Run `uv run ruff check yadc tests` and fix issues.
- Run `uv run basedpyright yadc/api`.
- Verify the full WebUI captioning flow end-to-end.
- Verify graceful shutdown under load (multiple SSE clients + active captioning job).
- Fix Ctrl+C shutdown with active SSE connections (dispatch `ShutdownEvent` before uvicorn waits for connections).
- Add a note to `todo.md` that a FastAPI migration is deferred until after this refactor is validated.

### Phase 4: Hardening — Shutdown Timeouts and Integration Tests

**Goal**: Make the server resilient to edge cases and add automated tests for the async behaviour.

- **Uvicorn graceful shutdown timeout**: Add `timeout_graceful_shutdown` to `uvicorn.Config` (e.g. 5–10 seconds) so that even if SSE generators or background tasks fail to exit, uvicorn force-cancels them and the process terminates.
- **SSE shutdown integration test**: Write a test that:
  1. Starts the Quart app via `quart.testing.QuartClient` or `httpx.AsyncClient`.
  2. Opens an SSE connection to `/api/events`.
  3. Simulates `SIGINT` (or calls the shutdown path directly).
  4. Asserts that the connection closes and the server terminates within the timeout.
- **Connection leak test**: Verify that client disconnect removes the queue from `SSEEvents._queues` (could be tested by inspecting the service directly or counting open listeners).
- **EventDispatcher bridge test**: Verify that `dispatch()` called from a background thread correctly schedules an async handler on the main loop.

**Verify**:
- `pytest tests/` passes including the new tests.
- `ruff` and `basedpyright` remain clean.

## File-by-File Reference

| File | Change Summary |
|---|---|
| `pyproject.toml` | `flask` → `quart`, `waitress` → `uvicorn[standard]` |
| `yadc/api/application.py` | Flask → Quart, uvicorn runner, remove manual SIGINT handler |
| `yadc/api/configuration.py` | Remove `http_threads` |
| `yadc/api/controllers/blueprints.py` | `quart.Blueprint` instead of `flask.Blueprint` |
| `yadc/api/controllers/__init__.py` | No change |
| `yadc/api/controllers/utils_json.py` | No change (Quart Response is compatible) |
| `yadc/api/controllers/models_errors.py` | No change |
| `yadc/api/controllers/app_frontend.py` | `async def` routes, `await quart.send_file/send_from_directory` |
| `yadc/api/controllers/api_events.py` | `async def` route + async generator, remove waitress refs |
| `yadc/api/controllers/api_datasets.py` | `async def` media/thumbnail routes, `await quart.send_file` |
| `yadc/api/controllers/api_*.py` (rest) | No changes |
| `yadc/api/modules/cors_middleware.py` | No changes |
| `yadc/api/modules/event_dispatcher.py` | Bridge async handlers onto the event loop |
| `yadc/api/modules/sse_events.py` | Full async rewrite: `asyncio.Lock`, `asyncio.Queue`, `async def receive` |
| `yadc/api/modules/job_scheduler.py` | No changes |
| `yadc/api/services/*.py` | No changes |
| `yadc/cli_webui.py` | Remove `--threads` option |

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **Quart + injector incompatibility** | Quart is a subclass of Flask concepts; injector's `@inject` works the same. We manually wire routes in closures, so FastAPI-style DI is not needed. |
| **Async event bridge deadlock** | `run_coroutine_threadsafe` is non-blocking from threads. `dispatch()` never awaits. |
| **Background captioning threads break** | They remain daemon threads. They call `event_dispatcher.dispatch()` (sync), which safely bridges to the loop. No changes to `CaptionJob`. |
| **Quart's `send_file` in sync routes** | We proactively make `send_file` routes `async def`. Other sync routes continue to work. |
| **Large PR** | Each phase is independently committable and leaves the app in a working state. |
| **Uvicorn static file serving differences** | We keep the same route table; Quart handles ASGI static responses. No mount changes needed. |

## Deferred Work (FastAPI Migration)

After Quart is validated, a future plan may migrate to FastAPI for:
- Native Pydantic request/response models (fixes the 21 basedpyright warnings in controllers).
- Automatic OpenAPI docs generation.
- Stronger typing on route parameters.

This is intentionally **out of scope** for this plan.

## Success Criteria

- [x] `yadc webui serve` starts with uvicorn.
- [x] All existing API endpoints return identical JSON shapes.
- [x] SSE connects, receives events, reconnects with `Last-Event-ID`, and resumption works.
- [x] Ctrl+C shuts down cleanly within 2 seconds even with active SSE connections.
- [x] `uv run pytest tests` passes (or at least no new failures introduced).
- [x] Full WebUI captioning flow works end-to-end.
- [ ] Uvicorn graceful shutdown timeout configured and honoured.
- [ ] Integration tests exist for SSE shutdown and EventDispatcher thread-to-async bridge.
