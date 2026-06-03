---
date: 2026-05-27
---

# Phase 1: Migrate from Flask/waitress to Quart/uvicorn

**Context:** The API backend was built on Flask + waitress (WSGI). SSE connections blocked threads indefinitely, causing Ctrl+C hangs, proxy zombie connections, and thread-pool exhaustion. The plan was to migrate to Quart + uvicorn (ASGI) as a stepping stone before a potential future FastAPI migration.

**Decision:** Swap the WSGI stack for ASGI with minimal controller churn.

**Rationale:**
- Quart is API-compatible with Flask — same Blueprints, `request`, `send_file`, `jsonify`.
- Only `send_file` routes and JSON-body routes needed `async def` changes.
- This validates async SSE works before committing to a larger FastAPI rewrite.

**What was done:**
- `pyproject.toml`: `flask` → `quart`, `waitress` → `uvicorn[standard]`, added `pytest-asyncio`.
- `yadc/api/application.py`: `Flask` → `Quart`, `waitress.serve()` → `uvicorn.run()`, removed manual `SIGINT` handler (uvicorn handles signals; `ShutdownEvent` dispatched in `try/finally`).
- `yadc/api/controllers/blueprints.py`: `flask.Blueprint` → `quart.Blueprint`.
- `yadc/api/controllers/app_frontend.py`: All routes `async def`; `await send_file` / `await send_from_directory`.
- `yadc/api/controllers/api_datasets.py`: Media/thumbnail routes `async def`; `await send_file`.
- All POST/PUT/PATCH controllers (`api_captioning`, `api_configs`, `api_datasets`, `api_envs`, `api_export`, `api_templates`): routes became `async def` and `await request.get_json()` (Quart requirement).
- `api_events.py`: Removed waitress-specific `client_disconnected` WSGI environ check.
- `cli_webui.py`: Removed `--threads` option.
- `configuration.py`: Removed `http_threads` field.
- Tests: Updated all API tests to use `Quart`, `pytest.mark.asyncio`, `await resp.get_json()`.

**Post-commit fix:**
- `api_events.py`: Quart's ASGI `Response` requires async iterables for streaming bodies. The original sync generator wrapped in `stream_with_context()` caused `TypeError: 'async for' requires an object with __aiter__ method, got generator`.
- Fixed by making `_retrieve_events()` an `async def` async generator and removing `stream_with_context`.
- Since `SSEEvents.receive()` is still synchronous (uses `threading.Condition`), wrapped `next(sync_gen)` calls in `asyncio.to_thread()` to avoid blocking the event loop thread. Added `typing.cast` for basedpyright compatibility.

**Verification:**
- `ruff check yadc tests` — clean.
- `basedpyright yadc/api` — 0 errors (warnings are pre-existing).
- `pytest tests/` — 128 passed, 14 skipped.
- Server boots successfully on uvicorn.
- Webui logs show no errors; SSE endpoint serves events correctly.

**Files touched:**
- `pyproject.toml`
- `yadc/api/application.py`
- `yadc/api/configuration.py`
- `yadc/api/controllers/blueprints.py`
- `yadc/api/controllers/app_frontend.py`
- `yadc/api/controllers/api_datasets.py`
- `yadc/api/controllers/api_captioning.py`
- `yadc/api/controllers/api_configs.py`
- `yadc/api/controllers/api_envs.py`
- `yadc/api/controllers/api_export.py`
- `yadc/api/controllers/api_templates.py`
- `yadc/api/controllers/api_events.py`
- `yadc/api/controllers/utils_json.py`
- `yadc/api/modules/cors_middleware.py`
- `yadc/cli_webui.py`
- `tests/api/test_captioning.py`
- `tests/api/test_configs.py`
- `tests/api/test_envs.py`
- `uv.lock`
