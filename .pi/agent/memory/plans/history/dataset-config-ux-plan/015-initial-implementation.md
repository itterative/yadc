---
date: 2026-05-30
---
# Initial Implementation: Dataset Edit Dialog with Config + Upload

**Context:** The `EditDatasetDialog` on the dataset listing page was a barebones raw TOML editor. The user wanted to extend it with the full structured config editor (Form/Advanced/History from the side panel) and add an Upload tab for appending more images to managed datasets.

**Implementation:**

### Phase 1: Extract shared component
- Moved `DatasetConfig.svelte` + `ConfigHistory.svelte` from `routes/datasets/[name]/` to `lib/components/datasets/`
- Updated `SidePanel.svelte` import path

### Phase 2: Backend append endpoint
- Added `append_dataset_from_upload()` to `DatasetUploadService`
  - Validates dataset exists and is managed (`source === 'upload'`)
  - Writes files to existing `images/` and `folders/` directories
  - Same validation as create upload (PIL verify, TOML parse, orphan sidecar check, flat-only)
  - If new top-level folders are uploaded, adds new `[[dataset]]` entries to config TOML using `tomlkit` (preserves existing config sections)
  - Calls `rescan_dataset()` to re-index
- Added `POST /datasets/<name>/upload` route to `api_datasets.py`
  - Streaming NDJSON response (same shape as create upload)
  - No name field in FormData (uses URL parameter)

### Phase 3–4: Frontend tabbed dialog + shared upload component
- Created `DatasetUploadPanel.svelte` — reusable upload UI with `mode: 'create' | 'append'`
  - `'create'` shows dataset name input
  - `'append'` uses provided `datasetName`
  - Encapsulates FileDropZone, progress bar, cancel button, error display
- Rewrote `UploadDatasetTab.svelte` as thin wrapper around `DatasetUploadPanel mode="create"`
- Rewrote `EditDatasetDialog.svelte` with PillTabs:
  - **Config tab**: `DatasetConfig` component (Form/Advanced/History)
  - **Upload tab**: `DatasetUploadPanel mode="append"` (only shown for `source === 'upload'`)
- Added `appendUploadDataset()` helper to `datasetImages.ts`

### Phase 5: Wire up listing page
- Updated `+page.svelte` to pass `source={editingDataset.source}` to `EditDatasetDialog`

**Files touched:**
- `yadc/webui/src/lib/components/datasets/DatasetConfig.svelte` (moved)
- `yadc/webui/src/lib/components/datasets/ConfigHistory.svelte` (moved)
- `yadc/webui/src/lib/components/datasets/DatasetUploadPanel.svelte` (new)
- `yadc/webui/src/lib/components/datasets/UploadDatasetTab.svelte` (rewritten)
- `yadc/webui/src/routes/datasets/[name]/SidePanel.svelte`
- `yadc/webui/src/routes/EditDatasetDialog.svelte` (rewritten)
- `yadc/webui/src/routes/+page.svelte`
- `yadc/webui/src/lib/stores/datasetImages.ts`
- `yadc/api/services/dataset_upload.py`
- `yadc/api/controllers/api_datasets.py`

**Tests:** All 192 tests pass. svelte-check 0 errors. Build succeeds.
