---
date: 2026-05-31
---
# Review Pass: Dataset Edit + Upload Implementation

**Context:** Review of all changes for the dataset-edit-and-upload-plan (Phases 1–5 complete + staging/conflict resolution + managed dataset lifecycle). Covers the `EditDatasetDialog` rewrite, append upload with conflict handling, managed dataset path guards, folder management, and image deletion.

## Summary

The implementation delivers a polished upload-and-manage experience for managed datasets. The staging/conflict flow is well-architected. Backend guards for managed dataset paths are thorough (both frontend disabled inputs and server-side rejection). Image deletion with sidecar cleanup works correctly. Below are findings from this review pass.

## Must Fix Before Merge

### 1. `commit_staged_upload` missing managed-source guard ✅

`append_dataset_from_upload` validates `info.source != "upload"`, but `commit_staged_upload` only checks for existence and config_path. If someone discovers the staging UUID, they could commit against a non-managed dataset.

**Fix:** Added `if info.source != "upload": raise ValueError(...)` at the top of `commit_staged_upload`.

**Files:** `dataset_upload.py` (`commit_staged_upload`)

### 2. `ConfigHistoryService._prune()` connection leak ✅

**Status:** Already fixed in current code. `_prune()` receives `conn` as a parameter from `save_snapshot()` and uses the same connection. No leak.

### 3. Config history orphaned on dataset deletion ✅

Same as finding #1 in the config-ux review — `unregister_dataset` doesn't clean up `config_history` rows.

**Fix:** Added `dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE` via migration step 6. `save_snapshot()` now inserts `dataset_id` so the FK can cascade on delete.

**Files:** `db_migrations.py`, `config_history.py`

## Should Fix

### 4. `_remove_folder_dataset_entries` has extra blank line ✅

**Fix:** Removed the extra blank line.

**File:** `datasets.py

### 5. History restore + folder deletion divergence (known gap from plan) ✅

When a folder is deleted via the Manage tab, the `[[dataset]]` entry is removed from config TOML. But restoring an older config history snapshot would re-add that entry even though the folder no longer exists on disk.

**Fix:** The `restore_config_history` endpoint now checks restored config entries for missing `folders/X` directories. If any are found, a `warnings` array is included in the response (e.g. `"Folder 'train' is referenced in config but no longer exists on disk."`).

**Files:** `api_configs.py` (`restore_config_history`)

### 6. Advanced TOML edit for managed datasets needs backend validation

The frontend shows a warning message in Advanced mode for managed datasets, but the backend `PUT` endpoint validates path changes. If a user edits a non-path field (e.g. changes `max_tokens`) in Advanced mode, the full TOML content is sent via PUT — the path check compares the original and new path sets and allows the save. This works correctly.

However, if a user manually edits the `[[dataset]]` paths to add a new relative path like `folders/hack` pointing outside the managed directory, the server accepts it (since the path *string* is different from the original set, the check catches it). Wait — actually the check rejects *any* path set change. So users can't even add valid paths via Advanced mode. This is correct behavior since managed paths should only be changed through upload.

**No fix needed** — just confirming the guard is correct.

## Acknowledged / No Action Needed

### 7. Managed dataset state directory location

The plan TODO mentions moving managed datasets to `STATE_PATH/datasets/<name>/` to avoid collisions with templates and keys. This is still unresolved. Currently, datasets live directly in `STATE_PATH/<name>/` alongside templates and other state files. This should be addressed before merge but is not a code quality issue — it's a structural migration.

### 8. `DatasetManageTab` is minimal but functional

The Manage tab currently only shows a Folders section. It's designed to be extended (e.g. adding image-level management, bulk operations). The current implementation is clean and doesn't over-reach.

### 9. `EditDatasetDialog` hides single tabs correctly

The `PillTabs` component's `hideSingle` prop is used — when `source !== 'upload'`, only the Config tab is shown, and the pill bar is hidden. Clean UX.

### 10. `ImageDetail.svelte` delete button placement

The delete button appears next to the Edit button in the image detail panel. This was flagged in the plan TODO as potentially confusing (users might think it deletes the caption, not the image). The confirmation dialog text is clear though: `Delete "filename" and its sidecars?`. Acceptable for now, but consider moving it to a kebab menu or adding a visual separator later.

## Positive Observations

1. **Staging folder approach** — Writing to `.staging/<uuid>/` first, detecting conflicts, then committing is clean and handles edge cases well (abort cleanup, stale staging dir cleanup, UUID collision avoidance).
2. **Grouped conflict resolution** — Sidecars follow the resolution of their image group (overwrite/skip/keep_both applies to image + all sidecars). This avoids confusing per-sidecar resolution.
3. **`_delete_file_with_sidecars`** — Correctly removes `.txt`, `.toml`, `.history~`, and `.*.draft~` sidecars matching the image stem. Handles draft naming convention (`IMAGE_STEM.DRAFT_NAME.draft~`).
4. **`_compute_delete_path`** — Computing the API-facing delete path on the backend (instead of frontend path parsing) is the right call. Frontend just uses `item.delete_path` directly.
5. **Relative path support** — Managed configs store paths like `images` and `folders/train` instead of absolute paths. Makes configs portable across machines. `resolve_dataset` and `_scan_dataset` resolve them correctly at read time.
6. **Backward-compatible path handling** — `delete_items` supports both prefixed paths (`images/foo.jpg`) and bare paths (`foo.jpg`) for backward compat.
7. **Test coverage** — 1156 lines of upload tests covering create, append, conflicts, commit resolutions, delete, list_folders, edge cases.
