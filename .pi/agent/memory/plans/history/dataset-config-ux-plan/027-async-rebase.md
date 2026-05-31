---
date: 2026-06-01
---
# Async Captioners Rebase

**Context:** The base branch (`feature/svelte-frontend`) added async captioners (commit `8ca1b89`) to enable cancellation of mid-flight captioning jobs. This feature branch (`feature/svelte-frontend-dataset-config-editing-improvements`) was branched off before that change. Rebasing was required to move the branch onto the async architecture.

**Backup branch:** `backup/dataset-config-ux-before-async-rebase`

## What changed in the base branch

- `CaptionJob` (threaded) → `AsyncCaptionJob` (`asyncio.Task`)
- `start_job()` / `stop_job()` / `get_status()` → `start_job_async()` / `stop_job_async()` / `get_status_async()`
- `_do_run()` / `_caption_one()` → `_ado_run()` / `_acaption_one()` (all async)
- `APICaptioner` now constructed via `await APICaptioner.create(..., async_session=...)`
- `_cleanup()` (threaded delay) → `_cleanup_async()` (async sleep + await rescan)
- `caption_single()` and `caption_single_async()` wrappers removed entirely

## Rebase conflicts and resolutions

### 1. `tests/api/test_captioning.py`
Conflict between the branch's `caption_single_async` mocks and the base's `caption_single` mocks.
**Resolution:** Adopted the base branch's async method names (`start_job_async`, `get_status_async`). Updated `test_caption_single_image_returns_job_info` to assert on `start_job_async` with `image_ids=[1]` instead of the removed `caption_single_async` wrapper.

### 2. `yadc/api/controllers/api_captioning.py` — removed `caption_single_async`
The branch had a `caption_single_async` wrapper that the base branch had already removed. The rebase re-introduced it.
**Resolution:** Deleted the `caption_single_async` method from `CaptioningService`. Inlined the single-image logic in the controller: `single_opts = options.model_copy(update={"image_ids": [image_id]})` → `await captioning.start_job_async(name, single_opts)`.

### 3. `yadc/api/services/captioning.py` — multiple conflicts
- **`is_captioning()` / `get_status()`**: The branch had added sync `is_captioning()` and `get_status()` methods. The base branch only had async methods. `api_datasets.py` calls `is_captioning()` from sync route handlers.
  **Resolution:** Kept `is_captioning()` as a **sync** method (reads `_async_jobs` dict, no lock needed for a simple alive check). `get_status_async()` remains async.

- **`_cleanup_async()` rescan**: The branch's `_cleanup()` did a threaded delay + `rescan_dataset()`. The base branch's `_cleanup_async()` only did cleanup without rescan.
  **Resolution:** Merged both — the async cleanup now `await`s `rescan_dataset(dataset_name)` at the end.

- **`expect_file_change` in `_acaption_one()`**: The branch had added file-level watcher registration in the non-draft path. The base branch had a different non-draft path without it.
  **Resolution:** Kept the branch's `expect_file_change` calls (caption, TOML, history paths) in `_acaption_one()` — critical for watcher suppression.

### 4. `tests/api/test_captioning_unit.py`
The branch's unit tests used the old sync `CaptionJob` and `CaptioningService._cleanup()` APIs.
**Resolution:** Updated the entire test file:
- Imports: `CaptionJob` → `AsyncCaptionJob`, added `AsyncMock`, `asyncio`
- `_mock_model()`: `model.predict.return_value = ...` → `model.predict = AsyncMock(return_value=...)`
- All `def test_*` → `async def test_*` with `@pytest.mark.asyncio`
- `CaptioningServiceCleanupRescan`: `_lock`/`_jobs` → `_async_lock`/`_async_jobs`, `patch("time.sleep")` → `patch("asyncio.sleep")`, `_cleanup()` → `await _cleanup_async()`

### 5. `tests/api/test_captioning_unit.py` — rebase re-introduced old sync imports
During a later rebase step, the old branch's sync imports (`threading`, `CaptionJob`) were re-introduced as conflict markers.
**Resolution:** Accepted the HEAD (already-async) versions of the imports.

## Memory updates

The following memories/docs were updated to reflect async APIs:
- `docs/captioning-workflow.md` — Model setup now shows `await APICaptioner.create(...)`
- `docs/dataset-watcher.md` — `CaptionJob._caption_one()` → `AsyncCaptionJob._acaption_one()`, `_cleanup()` → `_cleanup_async()`
- `plans/selective-image-refresh-plan.md` — All `CaptionJob` → `AsyncCaptionJob`, `_caption_one()` → `_acaption_one()`, `_do_run()` → `_ado_run()`
- `plans/history/selective-image-refresh-plan/004-active-image-indicator.md` — `_caption_one()` → `_acaption_one()`
- `plans/history/dataset-config-ux-plan/026-watcher-captioning-improvements.md` — `_caption_one()` → `_acaption_one()`, `_cleanup()` → `_cleanup_async()`

## Files touched

- `yadc/api/controllers/api_captioning.py`
- `yadc/api/services/captioning.py`
- `tests/api/test_captioning.py`
- `tests/api/test_captioning_unit.py`
