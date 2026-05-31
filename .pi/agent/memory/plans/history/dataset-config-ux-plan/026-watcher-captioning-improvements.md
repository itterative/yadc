---
name: dataset-config-ux-done
description: Completed todos from the dataset config UX plan. Kept for reference.
category: meta
---

# Dataset Config UX — Completed Todos

## Phase 1: Dataset Entries + Extras

- **Relative paths for uploaded datasets**: When `source === 'upload'`, display dataset entry paths relative to the dataset's state directory instead of showing the absolute state-folder paths.
- **Image/folder upload for managed datasets**: When `source === 'upload'`, allow uploading new images or folders to the dataset directly from the Paths & Variables section.
- **Path editing for external datasets**: `rescan_dataset()` now re-registers the filesystem watcher with the current config paths after scanning.
- **Extras value type support**: Auto-detects types from parsed config, type selector dropdown (str/num/bool/obj) next to each key.
- **TomlEditor (and JinjaEditor) doesn't use bindable for value**: refactored both to use `$bindable()` for `value`.
- **JinjaEditor should expose the template variables**: added `bind:variables` (bindable string array, synced via `$effect`).
- **Managed datasets should live in `STATE_PATH/datasets/` subfolder**: Datasets now live at `STATE_PATH/datasets/<name>/` via the `DATASETS_DIR` Path constant.
- **History restore should wait for confirmation**: The restore button calls the confirmation dialog.
- **Dataset deletion doesn't drop config revision history**: Added `delete_dataset_history()` method to `ConfigHistoryService` and call it from `unregister_dataset()`.
- **Dataset refresh notification**: Fixed regression caused by `_unwatch_dataset_locked()` clearing `_expected_sources` when `watch_dataset()` re-registers watches. Fix: save/restore `_expected_sources` across re-registration.
- **Edge case when deleting image/folder while captioning**: Backend returns 409 Conflict for destructive operations when a captioning job is active.
-  **ConfigHistoryService.save_snapshot should handle missing dataset_id gracefully**: Logging a warning if dataset_id is not found so we don't add orphan entries in the history.

## Watcher & Captioning Fixes

- **Notification dataset is refreshed when manually changing caption**: Implemented file-level expected-changes system in `DatasetWatcherService`. Backend registers expected file paths before writing. If all changes in a debounce window are expected, the event is tagged with `job_id = "self"` and suppressed by the frontend.
- **Captioning doesn't refresh draft**: After single-image captioning completes, `ImageDetail` always re-fetches full caption data (including drafts).
- **Captioning single image uses draft**: Caption button shows the draft name when configured (e.g. "Caption → my-draft"). Spinner shows "Generating draft…".
- **Refactor `_unwatch_dataset_locked` to preserve `_expected_sources`**: Addressed by the path-diff short-circuit (only calls `_unwatch_dataset_locked` when paths actually change) combined with the `saved_source` workaround for that case. The remaining idea of a `CaptioningService` callback for `get_active_job_id` is no longer needed — `_on_dataset_changed` now skips rescans for self-originated changes, making the `_expected_sources` cleanup timing non-critical.
- **Diff watched paths in `watch_dataset()` to avoid redundant re-registration**: Every `rescan_dataset()` call and `DatasetChangedEvent` triggers `watch_dataset()` with the same paths, causing unwatch→rewatch cycles. Now diffs the new paths against `_watches[dataset_name]` and short-circuits when unchanged. The `saved_source` workaround is still kept for the case where paths *do* change during captioning (e.g. `_refresh_stale_datasets` picks up a manually edited TOML).
- **Migrate captioning jobs from `expect_changes(job_id)` to file-level `expect_file_change()`**: Captioning jobs now register individual file writes via `expect_file_change()` in `_acaption_one()` before each write (caption, TOML, history, draft). This makes per-file suppression reliable regardless of the `_expected_sources` cleanup timing. `_expected_sources` is kept for job_id tagging on `DatasetChangedEvent` (frontend routing). The delayed cleanup remains but its timing is no longer critical — file-level matching is the primary suppression mechanism.
- **Skip redundant rescans for self-originated changes**: `_on_dataset_changed` now skips `rescan_dataset` when the `DatasetChangedEvent` carries a `job_id` (captioning or `"self"` webui edit). Per-image `refresh_image_index` calls already keep the DB up to date for those. Only truly external changes (job_id is `None` or `""`) trigger a full rescan. A final `rescan_dataset` runs in `CaptioningService._cleanup_async()` (5s after job ends) as a safety net.
