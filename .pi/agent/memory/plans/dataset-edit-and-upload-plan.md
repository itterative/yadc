---
name: dataset-edit-and-upload-plan
description: Extend the dataset edit dialog with a structured config editor (Form/Advanced/History) and upload support for adding more images to existing managed datasets.
status: In Progress
last_history: 9
category: meta
---

# Dataset Edit + Upload Plan

## Overview

The current `EditDatasetDialog` is a barebones raw TOML editor. We want to:
1. **Replace it with the full structured config editor** (Form/Advanced/History tabs) — reusing `DatasetConfig.svelte`
2. **Add an Upload tab** for managed datasets (`source === 'upload'`) — append images/folders to an existing dataset

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dialog vs page | Keep as dialog | Simplest, no routing changes |
| Config tab | Form / Advanced / History sub-tabs | Same as side panel, users already know it |
| Upload merge | Overwrite by filename | Matches how filesystem scanning works; user-selected |
| Non-managed datasets | Hide Upload tab entirely | Upload only makes sense for managed (uploaded) datasets |

## Implementation Phases

### Phase 1: Extract DatasetConfig.svelte to shared component ✅

Moved `routes/datasets/[name]/DatasetConfig.svelte` → `lib/components/datasets/DatasetConfig.svelte`.
- Also moved `ConfigHistory.svelte` (relative import from DatasetConfig)
- Updated `SidePanel.svelte` import path
- svelte-check passes

### Phase 2: Backend — Append upload endpoint ✅

Added `POST /datasets/<name>/upload` that appends files to an existing managed dataset.

**New method on `DatasetUploadService`:**
`append_dataset_from_upload(name: str, files: list[tuple[str, BinaryIO]]) -> AsyncGenerator[UploadProgressEvent, None]`

**Behavior:**
- Validate dataset exists and is managed (`source === 'upload'`)
- Write files to existing `STATE_PATH/<name>/images/` and `folders/`
- Same validation as create upload (PIL verify, TOML parse, orphan sidecar check, flat-only)
- If new top-level folders are uploaded, add new `[[dataset]]` entries to config TOML
- Call `rescan_dataset()` to re-index
- Return `UploadProgressEvent` streaming (same shape as create)

**Config TOML merging:**
- Read existing `config.toml`
- Parse with `tomlkit`
- Merge in new dataset entries (deduplicate paths)
- Write back

### Phase 3: Frontend — Update EditDatasetDialog ✅

Replaced `EditDatasetDialog.svelte` with a tabbed dialog:

```
┌─ Edit my-dataset ──────────────────────────────────┐
│  [Config] [Upload*]                               ✕ │
├────────────────────────────────────────────────────┤
│                                                      │
│  Config tab: DatasetConfig component                 │
│  (Form / Advanced / History sub-tabs)                │
│                                                      │
│  Upload tab: FileDropZone + progress + submit        │
│                                                      │
│  [*] only shown if source === 'upload'              │
└────────────────────────────────────────────────────┘
```

**Props:**
- `open`, `datasetName`, `source`, `onclose`, `onsaved`

**State:**
- `mode: 'config' | 'upload'`
- Upload state mirrors `UploadDatasetTab` but without name input

**On save/append success:**
- Call `onsaved()` so the parent listing refreshes

### Phase 4: Frontend — Extract shared upload component ✅

Created `DatasetUploadPanel.svelte` that wraps the upload logic:
- **Props:** `datasetName`, `isNewDataset` (controls whether name input is shown), `oncomplete(result)`, `onerror(error)`
- **Reuses:** `FileDropZone`, progress bar, cancel button
- **Used by:** `UploadDatasetTab` (create mode, shows name input) and `EditDatasetDialog` (append mode, no name input)

**API helper:**
- `appendUploadDataset(name, files, onProgress, onChunk, signal)` → calls `POST /datasets/<name>/upload`

### Phase 5: Wire up listing page ✅

Updated `+page.svelte` (dataset listing):
- Pass `source` from `DatasetInfo` to `EditDatasetDialog`
- Ensure `onsaved` refreshes the listing

## Key Files

| File | Role |
|------|------|
| `yadc/webui/src/routes/datasets/[name]/DatasetConfig.svelte` | Move to shared |
| `yadc/webui/src/lib/components/datasets/DatasetConfig.svelte` | New shared location |
| `yadc/webui/src/lib/components/datasets/DatasetUploadPanel.svelte` | Reusable upload UI |
| `yadc/webui/src/routes/EditDatasetDialog.svelte` | Tabbed edit dialog |
| `yadc/webui/src/routes/+page.svelte` | Listing page |
| `yadc/webui/src/lib/stores/datasetImages.ts` | New `appendUploadDataset` helper |
| `yadc/api/controllers/api_datasets.py` | New `POST /datasets/<name>/upload` route |
| `yadc/api/services/dataset_upload.py` | `append_dataset_from_upload` method |

## Open Questions / Deferred

- **Upload conflict handling** ✅ **DONE** — Staging folder approach implemented. Files are written to `.staging/<staging_id>/` first. Conflicts detected against live dirs. User resolves per-file (skip/overwrite/keep_both) before commit. Staging dirs auto-cleaned after 24h by a background job.

- **Managed dataset path editing** ✅ **DONE** — Frontend disables path editing for managed datasets; backend rejects path changes with 400.

- **Config history restore + folder deletion divergence** ✅ **TODO** — When a folder is deleted via the Manage tab, the corresponding `[[dataset]]` entry is removed from config TOML. However, restoring an older config history snapshot would re-add that `[[dataset]]` entry even though the folder no longer exists on disk. On restore, we should either: (1) warn the user about missing folders, (2) auto-create empty folders for re-added entries, or (3) prune missing-folder entries from the restored snapshot before applying it.

- **Advanced edit mode for managed datasets**: The Advanced TOML tab lets users edit raw config directly. For managed datasets, we must ensure paths stay within the managed state directory. Options: (1) validate on save and reject/correct illegal paths, (2) hide Advanced mode for managed datasets entirely, (3) show a warning banner in Advanced mode for managed datasets. Needs refinement.
