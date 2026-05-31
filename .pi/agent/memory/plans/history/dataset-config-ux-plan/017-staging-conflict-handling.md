---
date: 2026-05-31
---
# Upload Conflict Handling — Staging Implementation

**Design decision**: See `002-upload-conflict-design.md` for the full design rationale.

**Implementation summary:**

### Backend: `dataset_upload.py`

- Added `_unique_path()` helper for auto-renaming on `keep_both`
- `DatasetUploadService.__init__` now accepts optional `job_scheduler: JobScheduler` and registers a periodic cleanup job (every 1 hour, removes staging dirs older than 24h)
- `append_dataset_from_upload()` completely rewritten to use staging:
  1. Creates a unique staging dir: `STATE_PATH/<name>/.staging/<uuid>/`
  2. Writes validated files to `staging/images/` and `staging/folders/`
  3. Detects conflicts by checking if the same relative path exists in the live dirs
  4. If conflicts found: emits `phase: "conflicts"` with `staging_id` and conflict list (file, existing_size, new_size)
  5. If no conflicts: auto-commits immediately
  6. On validation/write errors: cleans up staging dir
- Added `commit_staged_upload()` public method:
  1. Accepts `resolutions: dict[str, str]` mapping relative file paths to `"overwrite" | "skip" | "keep_both"`
  2. Moves files from staging to live dirs based on resolutions
  3. Auto-renames with `_1`, `_2`, etc. suffix for `keep_both`
  4. Updates config TOML with new `[[dataset]]` entries (same dedup logic as before)
  5. Rescans dataset, cleans up staging dir
  6. Emits `phase: "complete"` with updated dataset info
- Added `_cleanup_staging_dirs()` private method:
  - Iterates all dataset state dirs, looks for `.staging/` subdirs
  - Removes any subdir with `mtime > 24h`
  - Removes empty `.staging/` dirs

### Backend: `api_datasets.py`

- Updated `POST /datasets/<name>/upload` docstring to mention `"conflicts"` phase
- Added `POST /datasets/<name>/staging/commit` endpoint:
  - Accepts JSON `{ staging_id, resolutions }`
  - Returns NDJSON stream with `complete` or `error` events

### Frontend: `datasetImages.ts`

- Extended `UploadProgressEvent` interface with `staging_id?: string` and `conflicts?: UploadConflict[]`
- Added `UploadConflict` interface (`file`, `existing_size`, `new_size`)
- Added `commitStagingUpload()` helper:
  - POSTs to `/api/datasets/<name>/staging/commit`
  - Same streaming NDJSON parsing pattern as `uploadDataset`/`appendUploadDataset`

### Frontend: `DatasetUploadPanel.svelte`

- Complete rewrite to support conflict resolution UI:
  - **Upload flow**: Same as before (FileDropZone → progress → submit)
  - **Conflict flow**: When `phase === "conflicts"` event arrives:
    - Shows a table of all conflicts (file name, existing size, new size, action dropdown)
    - Bulk actions: "Skip All", "Overwrite All"
    - Per-file actions: Overwrite / Skip / Keep Both
    - "Back" button cancels and returns to file selection
    - "Apply" button calls `commitStagingUpload()`
  - **Commit flow**: Shows spinner while committing, then completes
- Added `stagingId`, `conflicts`, `conflictResolutions`, `isCommitting` state
- Default resolution for all conflicts is `"overwrite"`

### Tests

- All 192 backend tests pass (existing upload tests unaffected — they test `create_dataset_from_upload`)
- Frontend: svelte-check 0 errors, eslint passes, build succeeds

**Files touched:**
- `yadc/api/services/dataset_upload.py`
- `yadc/api/controllers/api_datasets.py`
- `yadc/webui/src/lib/stores/datasetImages.ts`
- `yadc/webui/src/lib/components/datasets/DatasetUploadPanel.svelte`
