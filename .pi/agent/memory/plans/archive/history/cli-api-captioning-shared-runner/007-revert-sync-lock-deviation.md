---
date: 2026-06-04
---
# Revert Phase 3 deviation #1: sync lock → async Protocol

**Context:** Phase 3 history entry `003-phase-3-async-caption-job-migration.md` documented a deviation from the original `asyncio.Lock` design: the runner's `CaptioningCallbacks` Protocol methods were made `def` (sync) to match the per-call callback model, which forced `AsyncCaptionJob`'s state helpers to use `threading.Lock` and made the state-update methods sync. The deviation note called this out for re-evaluation: "threading.Lock + sync state updates in an asyncio context is a deviation from the project's typical async style."

**Decision:** Revert the deviation. Make the `CaptioningCallbacks` Protocol methods `async def` and use `asyncio.Lock` for `AsyncCaptionJob`'s state. The right design was the original one.

**Why this is better:**

- **Project convention is async.** Other services in the API (`DatasetService`, `EventDispatcher`, etc.) use `asyncio.Lock` for state. The sync lock was a one-off deviation with no good reason once the Protocol is the right shape.
- **No `threading.Lock` in the asyncio loop.** The original deviation was mechanically correct (the lock is held for nanoseconds, the GIL makes Python int increments atomic, `threading.Lock` blocks the event-loop thread which is fine) but stylistically wrong for a single-threaded async codebase.
- **Callbacks can do real async work.** With `async def` callbacks, implementations can `await asyncio.Lock`, `await aiohttp`, or any other async primitive directly. With sync callbacks, async work had to be scheduled via `asyncio.create_task` (which the Protocol's docstring suggested). The async Protocol is the simpler, more flexible shape.
- **No race window at all.** With `asyncio.Lock`, the "tiny race window" concern from the deviation note goes away. The lock is held for a few coroutine steps; the state is always consistent.

**Code changes:**

- `yadc/core/captioning/runner.py` — `CaptioningCallbacks.on_token` / `on_image_started` / `on_image_captioned` / `on_image_error` are now `async def`. The runner does `await callbacks.on_X(...)` for each call.
- `yadc/api/services/captioning.py` — `AsyncCaptionJob` callback methods are now `async def`. `threading.Lock` → `asyncio.Lock`. `_set_state` / `_emit_status` / `_snapshot_locked` are now `async def` (they all touch the lock and are awaited from the callback methods and from `_arun` / `_ado_run` / `start_job_async`). `start_job_async` awaits the `job._set_state(total=len(to_do))` call instead of sync. The `import threading` line is removed (no other uses).
- `yadc/cli_caption.py` — `CLICallbacks` methods are now `async def` (Protocol match). Bodies unchanged.
- `tests/core/captioning/test_runner.py` — `MagicMock(spec=CaptioningCallbacks)` → `AsyncMock(spec=CaptioningCallbacks)`. `AsyncMock` spec'd against the Protocol creates `AsyncMock` instances for `async def` methods, which the runner can `await`. All existing call assertions (`assert_called_once_with`, `assert_has_calls`, `assert_not_called`) work unchanged.
- `tests/api/test_captioning_unit.py` — `TestCaptioningCallbacks` tests are now `async` (with `@pytest.mark.asyncio`) and `await` the callback methods. `TestExpectedChangeRegistrar` and `TestAdoRunWithRunner` are unchanged (the registrar method itself is still sync, and `_ado_run` was already async).

**Verification:** All 386 tests pass. Ruff + basedpyright clean. End-to-end smoke test (`uv run yadc caption test_pedro.dataset --no-stream --env integration-tests-local-llamacpp`) works; the CLI's `CLICallbacks` is now async but the interactive action menu is unchanged from the user's perspective.

**Out of scope:**

- The "threading.Lock in async" discussion in `003-phase-3-async-caption-job-migration.md` is now stale. Future agents reading that history entry should also read this one — the deviation it documents is reverted, and the lock is back to `asyncio.Lock`.
- The Protocol's docstring (in `runner.py`) was updated to reflect the async design; the rest of the runner doc is unchanged.
