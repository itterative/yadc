---
date: 2026-05-27
---

# Phase 3: Cleanup, Testing, and Hardening

**Context:** Phases 1 and 2 completed the Quart/uvicorn migration and made SSE fully async. Phase 3 was intended as a cleanup pass, but during verification we discovered and fixed a critical shutdown bug.

## Ctrl+C shutdown fix (the big one)

**Problem:** With active SSE connections, pressing Ctrl+C caused uvicorn to hang indefinitely on "Waiting for connections to close." A second Ctrl+C was required to force-quit.

**Root cause:** Uvicorn's shutdown sequence is:
1. Call `connection.shutdown()` on all HTTP connections
2. Wait for all connections to close naturally
3. Only THEN run `lifespan.shutdown` (which triggers Quart's `after_serving`)

Since SSE connections never close on their own, step 2 blocked forever. Our `after_serving` handler (which dispatched `ShutdownEvent`) was never reached.

**Fix:** Monkey-patch `uvicorn.Server.handle_exit` to dispatch `ShutdownEvent` *immediately* when SIGINT/SIGTERM is received, before uvicorn starts waiting for connections. This sets `SSEEvents._shutdown = True`, causing all SSE generators to exit within their 1-second timeout window. Uvicorn then sees the connections close and completes graceful shutdown.

- `yadc/api/application.py`:
  - Switched from `uvicorn.run()` to explicit `uvicorn.Config` + `uvicorn.Server` + `server.run()`
  - Added `_handle_exit` wrapper around `server.handle_exit` that dispatches `ShutdownEvent` before chaining to uvicorn's handler
  - Removed the `finally` block in `run()` (no longer needed)
  - Removed the `after_serving` handler (redundant with the signal handler, and would run too late anyway)
  - Added `lifespan="on"` to uvicorn config to ensure lifespan events are always used

**Verification:**
- Start server, connect 2 SSE clients, send SIGINT → server exits cleanly within ~1.5 seconds
- Logs show: `Shutdown event received, stopping 2 listeners` → `Shutting down` → `Application shutdown complete` → `Finished server process`

## Remaining cleanup

- Removed all remaining "Flask" references in docstrings and comments:
  - `yadc/api/controllers/utils_json.py:37` — "Flask Response" → "Quart Response"
  - `yadc/api/modules/cors_middleware.py:22` — "Flask blueprint" → "Quart blueprint"
  - `yadc/webui/src/lib/api.ts:3-4` — "Flask" → "Quart/uvicorn" in frontend API comments
- Removed unused `import json` from `tests/api/test_captioning.py` and `tests/api/test_configs.py` (found by ruff).
- Updated `todo.md`:
  - Marked "Convert API backend to async (ASGI)" as **DONE**.
  - Added a "Deferred: FastAPI migration" note.
  - Marked "WSGI thread exhaustion" under SSE resumption as **DONE**.

## Verification

- `ruff check yadc tests` — clean (0 errors).
- `basedpyright yadc/api` — 0 errors, 50 warnings (all pre-existing, none new).
- `pytest tests/` — 128 passed, 14 skipped.
- Server boots cleanly on uvicorn, frontend loads, SSE connects, Ctrl+C exits cleanly.
- No `flask` or `waitress` imports/references remain in `yadc/` or `tests/` source.

**Files touched:**
- `yadc/api/application.py`
- `yadc/api/controllers/utils_json.py`
- `yadc/api/modules/cors_middleware.py`
- `yadc/webui/src/lib/api.ts`
- `tests/api/test_captioning.py`
- `tests/api/test_configs.py`
- `.pi/agent/memory/todo.md`
