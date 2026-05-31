---
date: 2026-05-31
---
# Review Pass: Dataset Config UX Implementation

**Context:** Comprehensive review of all changes between `feature/svelte-frontend` and `feature/svelte-frontend-dataset-config-editing-improvements` (74 files, +10,640/-5,461 lines). All 215 tests pass, svelte-check clean, eslint clean, build succeeds.

## Summary

The implementation is well-structured and comprehensive. The code follows established patterns, has good test coverage, and the architecture decisions are sound. Below are findings organized by severity.

## Must Fix Before Merge

### 1. Config history not cleaned up on dataset deletion ✅

`DELETE /configs/<name>` → `datasets.unregister_dataset()` removes the state directory but never deletes rows from `config_history`. Orphaned rows accumulate in SQLite with no cleanup path.

**Fix:** Added `dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE` to `config_history` via migration step 6. Foreign key cascade ensures history rows are auto-deleted when the dataset is removed. Also updated `save_snapshot()` to look up and store `dataset_id`.

**Files:** `db_migrations.py` (migration 6), `config_history.py`

### 2. `ConfigHistoryService` connection leaks in `save_snapshot()` ✅

**Status:** Already fixed in current code. `_prune()` receives `conn` as a parameter and operates on the same connection opened by `save_snapshot()`. No second connection is opened. Other methods (`list_history`, `get_entry`, `delete_entry`) each open and close their own connection properly via `try/finally`.

**Files:** `config_history.py` — no change needed.

### 3. `commit_staged_upload` missing managed-source guard ✅

`append_dataset_from_upload` correctly checks `info.source != "upload"`, but `commit_staged_upload` only checks `info is None or not info.config_path`. A malicious client could theoretically call commit directly on a non-managed dataset's staging dir if they know the staging UUID.

**Fix:** Added `if info.source != "upload": raise ValueError(...)` to `commit_staged_upload`.

**Files:** `dataset_upload.py` (`commit_staged_upload`)

## Should Fix

### 4. Inconsistent TOML library usage in `dataset_upload.py` ✅

`create_dataset_from_upload()` uses `toml.dump()` for initial config write, while `commit_staged_upload()` uses `tomlkit` for appending entries.

**Fix:** Replaced `toml.dump(raw, f)` with `tomlkit.dump(raw, f)`. Removed `import toml` entirely. Also replaced the remaining `toml.loads` in `_validate_toml()` with `tomlkit.loads`.

**Files:** `dataset_upload.py`

### 5. `create_dataset_from_upload` writes relative paths but `create_dataset` writes absolute paths ✅

**Fix:** Added a comment near the config write explaining that managed datasets use relative paths intentionally for portability, resolved against `config.toml`'s directory at scan time.

**Files:** `dataset_upload.py`

### 6. Frontend `DatasetConfig.svelte` — `populateFields` re-fetches after every save ✅

`handleSave()` calls `populateFields()` with the server response, then `handleConfigSaved()` calls `loadConfig()` again. This means every save triggers two fetches.

**Fix:** Split into two functions: `handleConfigSaved()` (normal saves — no `loadConfig`) and `handleHistoryRestored()` (history restore — calls `loadConfig`). Pass `handleHistoryRestored` to `ConfigHistory`.

**Files:** `DatasetConfig.svelte`

### 7. **BUG:** `ConfigHistory.svelte` — `confirmDialog.danger()` not awaited ✅

`confirmDialog.danger()` returns a `Promise<boolean>`. Without `await`, it always evaluates to truthy.

**Fix:** Added `await` before `confirmDialog.danger(...)`.

**Files:** `ConfigHistory.svelte`

### 8. `DatasetUploadPanel.svelte` — cancel button during commit

No issue — the abort controller is correctly reused.

### 9. `api_configs.py` — `_extract_dataset_paths` called with `toml_to_plain()` on every managed check

No action needed — configs are small.

## Acknowledged / No Action Needed

### 10. `datasets.py` — `delete_items` bare path fallback has subtle ordering

The bare path fallback tries `images_dir/rel`, then `folders_dir/rel` (as file), then `folders_dir/rel` (as dir). If a folder named "images" exists inside `folders/`, it would be addressed as `folders/images` via the prefixed path, not confused with the root `images/` dir. The `rel == MANAGED_IMAGES_PREFIX` guard correctly prevents deleting the root images folder. This is well-designed.

### 11. Staging cleanup background job is tied to `DatasetUploadService` lifecycle

If `job_scheduler` is not provided (e.g. in tests), the cleanup job isn't registered. This is the correct behavior — tests don't want background cleanup. Production always provides it via DI.

### 12. Frontend `configs.ts` — `fetchConfigHistory` is debounced

No action needed — low risk.

### 13. `SidePanel.svelte` — `onimagedelete` is threaded through but `SidePanel` doesn't use it directly

`SidePanel` passes `onimagedelete` to `ImageDetail` (via `ondelete` prop). The prop is correctly named differently at each level to avoid confusion. Clean design.

### 14. `+page.svelte` — `handleImageDelete` reloads datasets list after single image deletion

After deleting a single image, `handleImageDelete()` calls `loadInitial()` (reloads images) AND fetches datasets (to update image counts). This is correct — the image count badge on the topbar needs updating.

## Positive Observations

1. **Clean extraction of shared components** — `CaptionOptionsFields`, `KeyValueEditor`, `ConfigHistory`, `DatasetUploadPanel` are well-factored with clear props and bindable interfaces.
2. **Robust managed dataset guard** — Both frontend (disabled inputs) and backend (path change rejection) enforce managed dataset constraints.
3. **Good conflict resolution UX** — The staging → conflicts → commit flow with per-file resolution (skip/overwrite/keep_both) is comprehensive.
4. **Well-tested** — 1156 lines of upload service tests covering create, append, conflicts, commit, delete, list_folders. 228 lines of dict_utils tests.
5. **Consistent naming** — `MANAGED_IMAGES_PREFIX` / `MANAGED_FOLDERS_PREFIX` constants make path handling clear.
6. **Relative path support** — Managed configs use relative paths for portability, resolved at scan time. External configs use absolute paths. Clean separation.
7. **History diff view** — Using `@codemirror/merge` with `collapseUnchanged` for incremental diffs is a good UX choice.
