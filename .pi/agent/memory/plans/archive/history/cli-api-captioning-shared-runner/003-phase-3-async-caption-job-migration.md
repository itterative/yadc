---
date: 2026-06-04
---
# Phase 3: AsyncCaptionJob migration to CaptioningRunner

> **NOTE — significant design decision worth re-evaluating:** Deviation #1 below changes the state lock from `asyncio.Lock` to `threading.Lock` and makes the state-update methods sync instead of async. The motivating reason is mechanical (the Protocol's callbacks are sync), but threading.Lock + sync state updates in an asyncio context is a deviation from the project's typical async style. The race window is tiny (int increment / list append) and the existing tests pass, but a reviewer should sanity-check this — see "Re-evaluation" at the bottom of this section for follow-up considerations.

**Context:** Phase 3 of the shared-runner plan refactors `yadc/api/services/captioning.py::AsyncCaptionJob` to delegate the model-create / stream / save loop to `yadc.core.captioning.CaptioningRunner`. The plan called for `AsyncCaptionJob` to implement the `CaptioningCallbacks` Protocol and pass `self` to the runner.

**Decision:** Six small deviations from the plan, each documented below.

## 1. Lock type change: `asyncio.Lock` → `threading.Lock`

The plan didn't explicitly address lock choice, but the Protocol methods are **sync** (`def`, not `async def`). The original code used `asyncio.Lock` to guard the state, with state updates happening inside `async` methods. After Phase 3, state updates happen inside the Protocol's sync methods, which can't `await` an `asyncio.Lock`.

**Switched to `threading.Lock`.** The lock is held for nanoseconds (int increment, list append). The lock is acquired from the event-loop thread; the only contention is `snapshot()` being called from another task. `threading.Lock` blocks the thread until acquired, which is fine in a single-threaded asyncio loop.

**Knock-on effect:** `_set_state`, `_increment_processed`, `_record_error`, and `_emit_status` are now sync methods. `snapshot()` stays async for backward compat (it's called from `get_status_async` and `_cleanup_async` via `await`).

**Re-evaluation:** The deviation is mechanical and motivated by the Protocol's sync method signatures, but the codebase's other services (e.g. `DatasetService`, `EventDispatcher`) keep their state in pure-Python without locks. Follow-up considerations:

- **Is the lock actually needed?** The state is updated only from `_ado_run`'s event-loop-thread execution, and the only concurrent reader is `snapshot()` (also called from the event loop). Without a lock, Python's GIL makes the int increments atomic, but `self._error_messages.copy()` is not. If a reader sees a partial state (a list that's mid-append), it gets a consistent list, not a corrupted one. So the lock is technically only needed to prevent reading a stale-but-consistent list during append — a minor issue.
- **Could the state be simpler?** If we accept that "best-effort" state is fine for `JobInfo` snapshots (clients see a slightly delayed counter during a very brief window), the lock could be removed entirely.
- **Was an async-to-sync migration avoidable?** The alternative is making the Protocol's methods `async def` — but that's a deviation from the original Plan Phase 2 design, which picked `Protocol` specifically so `AsyncCaptionJob` could pass `self` synchronously. Changing the Protocol now would undo Phase 2's user-requested design.

The decision stands for Phase 3, but the reviewer should confirm: (a) threading.Lock is acceptable in async code, (b) the project doesn't have a convention against sync methods on otherwise-async services, and (c) the small race window is acceptable.

## 2. Extracted `_expected_change_registrar` as a method

The plan said: "`expected_change_registrar` closure calls `self._dataset_watcher.expect_file_change(self._dataset_name, path)` for each path before writes." It was implemented as an inline closure inside `_ado_run`.

**Extracted to a method** so it can be unit-tested directly. The closure was trivial (3 lines), so making it a method costs nothing and gives us a clean unit test target.

## 3. Stored parsed `Config` on `self._config` in `preflight_images`

The plan said: "Replace `preflight_images` body with a call to `load_dataset_config()`. Keep its signature." `preflight_images` returns `list[DatasetImage]`, but `_ado_run` now needs the parsed `Config` to build the runner.

**Stored the config on `self._config`.** The loader returns `(Config, list[DatasetImage])`; we keep both. `preflight_images` still returns just the list (signature preserved, existing tests pass), and `_ado_run` reads `self._config` after calling `preflight_images`.

## 4. Deleted local `CaptionJobOptions`, `apply_config_overrides`, `resolve_template`

The plan said (in Phase 5): "Delete the duplicated `apply_config_overrides` / `resolve_template` re-exports in the API (keep only `CaptionJobOptions` re-export for backward compat)." This implicitly put the deletion in Phase 5.

**Deleted them in Phase 3 instead**, because nothing in the API uses them anymore — the job delegates to `load_dataset_config` which has its own copies. The "keep only `CaptionJobOptions` re-export" line implied the re-export would still exist; instead, the controller and `services/__init__.py` now import `CaptionJobOptions` directly from `yadc.core.captioning`. No re-export needed, no basedpyright `reportPrivateLocalImportUsage` warning about importing from a non-exporting module.

Phase 5 becomes a no-op for these (just removes dead re-exports, but there are none).

## 5. `# pyright: ignore[reportUnusedParameter]` on `on_token`

The Protocol's `on_token(self, token: str)` is required by signature, but the API doesn't use the token. The basedpyright `reportUnusedParameter` rule fires on the unused `token` parameter.

**Used `# pyright: ignore[reportUnusedParameter]`** per project preference — keep the parameter name (for Protocol match) and explicitly suppress the unused-parameter warning. `del token` would have worked too but pollutes the body; an underscore-prefixed `_token` would break the Protocol match.

## 6. Test rewrites

The plan said: "Run API tests. `tests/api/test_captioning.py` and `tests/api/test_captioning_unit.py` should still pass (the API surface is unchanged)."

**`test_captioning.py` (5 tests) passed unchanged.** It only tests the public API surface (`CaptioningService.start_job_async` / `get_status_async`), which is preserved.

**`test_captioning_unit.py` (19 tests pre-refactor → 24 tests post-refactor) had two obsolete test classes removed and three new ones added:**

- `TestCaptionOneRegistersExpectedFiles` (6 tests) — deleted. These tested `_acaption_one` directly, which no longer exists. The runner's `caption_image` save + `expected_change_registrar` behaviour is already covered by `tests/core/captioning/test_runner.py`.
- `TestOverwriteFilter` (3 tests) — deleted. These tested the overwrite/draft/image_ids filter via `_ado_run`. The filter is now in `load_dataset_config`, covered by `tests/core/captioning/test_loader.py`.
- `TestCaptioningCallbacks` (10 tests) — new. Verifies the Protocol implementation: each of the 4 callback methods dispatches the right event(s), updates the right state, and handles edge cases (unknown path, empty state).
- `TestExpectedChangeRegistrar` (2 tests) — new. Verifies the closure-method calls `dataset_watcher.expect_file_change` for each path.
- `TestAdoRunWithRunner` (7 tests) — new. Verifies `_ado_run` builds a `CaptioningRunner` with the right args, iterates images, sets done/cancelled status correctly, continues after per-image errors, and re-raises `CancelledError`.

Test fixture `job` is unchanged. `_mock_model`, `_mock_settings`, `image_with_file`, `_fake_stream` deleted (used only by the obsolete tests). `_image_info` helper added for the new tests.

**Files touched:** `yadc/api/services/captioning.py` (substantial rewrite — `_ado_run` now uses the runner; `preflight_images` uses the loader; `CaptioningCallbacks` Protocol methods; lock type change; deletion of `_acaption_one` / `_apply_overrides` / `_resolve_template` / local `CaptionJobOptions` / local `apply_config_overrides` / local `resolve_template`; `_set_state` and `_emit_status` now sync), `yadc/api/services/__init__.py` (import `CaptionJobOptions` from new location), `yadc/api/controllers/api_captioning.py` (import `CaptionJobOptions` from new location), `tests/api/test_captioning_unit.py` (delete obsolete tests, add new ones), `.pi/agent/memory/plans/cli-api-captioning-shared-runner-plan.md` (status, phases_complete, last_history).
