---
name: image-upload-dataset-creation-plan
description: Allow users to upload images/folders from the browser when creating a new dataset via the WebUI.
---

# Image Upload for Dataset Creation

## Current State

- **Creating datasets** via the WebUI requires entering **filesystem paths** to image directories in a textarea (one path per line). This only works if images are already on the same filesystem as the server.
- **Importing datasets** requires an existing TOML config path on the server filesystem.
- `DatasetService.create_dataset()` takes a name + list of directory paths, writes a minimal TOML to `STATE_PATH/<name>/config.toml`, then scans those directories.
- Backend uses **Quart** (async Flask-compatible) — supports `await request.files` for multipart form data.

## Proposed Feature

Add an **"Upload Images"** tab to `AddDatasetDialog` that lets users select image files and/or folders from the browser. Images are uploaded to the server, stored in a dataset-specific directory under `STATE_PATH`, and a dataset is created from them.

## Architecture

### Storage Layout

Uploaded images are stored in:

```
STATE_PATH/<dataset-name>/data/
```

This coexists alongside `STATE_PATH/<dataset-name>/config.toml` — the TOML points to the `data/` directory as its image path. Deleting a dataset (unregister) already removes the entire state dir, so uploaded files are cleaned up automatically.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Upload storage location | `STATE_PATH/<name>/data/` | Self-contained; dataset unregistration already deletes state dir |
| Upload limit | ~500 MB, configurable | Prevents abuse; reasonable for most image datasets |
| Deduplication | None (overwrite by filename) | Simple; matches how filesystem scanning works today |
| Folder upload | `webkitdirectory` attribute | Supported by all modern browsers; gives full directory structure |
| Progress tracking | `XMLHttpRequest.upload.onprogress` | `fetch()` doesn't support upload progress natively |
| Name collision | Reject with error if dataset name already exists | Same behavior as current create/import flows |

## Files to Modify

| File | Changes |
|------|---------|
| `yadc/api/controllers/api_datasets.py` | Add `upload_dataset()` route — multipart handler |
| `yadc/api/services/datasets.py` | Add `create_dataset_from_upload()` method |
| `yadc/api/configuration.py` | Add `max_upload_size_bytes` field |
| `yadc/webui/src/routes/AddDatasetDialog.svelte` | Add "Upload" tab with file picker + progress |
| `yadc/webui/src/lib/stores/datasetImages.ts` | Add `uploadDataset()` helper |

## Implementation Phases

### Phase 1: Backend upload endpoint

- Add `max_upload_size_bytes` to `Configuration` dataclass.
- Add `create_dataset_from_upload(name, files)` to `DatasetService`:
  - Creates `STATE_PATH/<name>/data/` directory.
  - Saves each uploaded file (validate extension against `IMAGE_EXTENSIONS`).
  - Writes minimal TOML with `[[dataset]] path = "<data_dir>"`.
  - Calls `_register()` to index and return `DatasetInfo`.
- Add `POST /api/datasets/upload` route to `api_datasets.py`:
  - Accepts `multipart/form-data` with `name` (text) + `files` (file(s)).
  - Validates total size against `max_upload_size_bytes`.
  - Returns `DatasetInfo` as JSON (201) or error (400/413/409).
- Test with `curl -F name=... -F files=@...`.

### Phase 2: Frontend upload UI

- Add `uploadDataset(name, files, onProgress?)` to `datasetImages.ts`:
  - Sends `multipart/form-data` via `XMLHttpRequest` (for progress events).
  - Returns `Promise<DatasetInfo>`.
- Add "Upload" tab to `AddDatasetDialog.svelte`:
  - Dataset name input (shared field, or tab-local).
  - File input: dual-mode — individual files (`multiple`) + folder (`webkitdirectory`).
  - Selected file list with name + size preview.
  - Total file count / size display.
  - Upload progress bar.
  - Error handling (size limit, network errors, duplicate name).

### Phase 3: Polish

- Cancel upload support (abort `XMLHttpRequest`).
- Drag-and-drop zone on the upload tab.
- Validate file types client-side before upload (check extensions).

## API Contract

### `POST /api/datasets/upload`

**Content-Type**: `multipart/form-data`

**Fields**:
- `name` (text, required): dataset name
- `files` (file(s), required): one or more image files

**Success (201)**: `DatasetInfo` JSON (same shape as other dataset endpoints)

**Errors**:
- `400`: missing name, no files, non-image file extension, empty name
- `409`: dataset name already exists
- `413`: total upload size exceeds limit
