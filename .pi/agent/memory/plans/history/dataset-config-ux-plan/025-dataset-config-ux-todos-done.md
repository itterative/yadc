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

## Watcher & Captioning Fixes

- **Notification dataset is refreshed when manually changing caption**: Implemented file-level expected-changes system in `DatasetWatcherService`. Backend registers expected file paths before writing. If all changes in a debounce window are expected, the event is tagged with `job_id = "self"` and suppressed by the frontend.
- **Captioning doesn't refresh draft**: After single-image captioning completes, `ImageDetail` always re-fetches full caption data (including drafts).
- **Captioning single image uses draft**: Caption button shows the draft name when configured (e.g. "Caption → my-draft"). Spinner shows "Generating draft…".
