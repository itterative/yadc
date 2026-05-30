---
name: image-upload-dataset-creation-plan
description: Upload images/folders from the browser when creating a dataset via WebUI.
status: Complete (phases 1–13, 5b); phase 14+ deferred
last_history: 16
category: meta
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

Uploaded files are stored in:

```
STATE_PATH/<dataset-name>/
  config.toml
  images/          ← root files (flat)
  folders/         ← folder uploads, one subdir per top-level folder
    train/
    val/
```

Root files (selected via "Choose Files" or drag-dropped individually) go into `images/` flat. Folder uploads preserve their structure under `folders/`. The generated `config.toml` has one `[[dataset]]` entry for `images/` and one per top-level folder inside `folders/`. Deleting a dataset (unregister) already removes the entire state dir, so uploaded files are cleaned up automatically.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Upload storage layout | `images/` (flat) + `folders/` (structured) | Separates root files from folder uploads; each top-level folder gets its own `[[dataset]]` entry |
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

### Phase 1: Backend upload endpoint ✅

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

**Key details implemented:**
- Path sanitization on uploaded filenames via `resolve()` + `relative_to()` defense-in-depth — any path that resolves outside the dataset `data/` directory is rejected (catches `..`, absolute paths, symlinks, and normalization tricks).
- Preserves relative directory structure from `webkitdirectory` uploads.
- Name collision check returns 409 before any file I/O.
- Size limit checked via `request.content_length` (best-effort upper bound).
- All error responses use standardized `ErrorCode` values (`BAD_REQUEST`, `PAYLOAD_TOO_LARGE`, `CONFLICT`).

### Phase 2: Frontend upload UI ✅

- Created `yadc/webui/src/lib/upload.ts` — minimal fetch-like Promise wrapper around `XMLHttpRequest` with progress tracking and `AbortSignal` support.
- Added `uploadDataset(name, files, onProgress?)` to `datasetImages.ts`:
  - Builds `FormData` with `webkitRelativePath` preserved for folder uploads.
  - Uses `upload()` utility, calls `apiErrorMessage()` on failure.
  - Returns `Promise<DatasetInfo>`.
- Added "Upload" tab to `AddDatasetDialog.svelte`:
  - Dataset name input (tab-local, consistent with import/create tabs).
  - Dual file inputs: "Choose Files" (`multiple`, `accept="image/*"`) + "Choose Folder" (`webkitdirectory`).
  - Selected file list with name, size, and per-file remove button.
  - Total file count / size summary with "Clear all".
  - Upload progress bar with loaded/total bytes.
  - Error handling via existing dialog `error` banner.
- Updated `apiErrorMessage()` to accept a `ResponseLike` interface so it works with both native `fetch` `Response` and `UploadResponse`.
- Added `PAYLOAD_TOO_LARGE` to `ErrorCode` enum and 413 user-friendly message to frontend.

**Key details implemented:**
- Folder uploads preserve directory structure via `file.webkitRelativePath` passed as the third `FormData.append()` argument.
- Progress bar updates reactively during upload.
- All buttons have `type="button"` to prevent accidental form submission.
- File list uses `{#each ... (file.name + '-' + i)}` for stable keys.

### Phase 3: Polish

#### Drag-and-drop zone ✅

- Replaced the plain file picker buttons with a bordered drop zone area.
- Added `dragenter`/`dragover`/`dragleave`/`drop` handlers on the zone.
- Used a `dragCounter` to handle nested drag events correctly (prevents flicker when dragging over child elements).
- Added visual feedback: border and background color change to accent color on drag-over.
- Implemented recursive directory traversal via `webkitGetAsEntry()` / `FileSystemDirectoryReader`:
  - `readDirectoryEntries()` wraps `reader.readEntries()` in a Promise.
  - `collectFilesFromEntry()` recursively walks directories, collecting all files.
  - `getFileFromEntry()` wraps `entry.file()` in a Promise.
  - Preserves relative paths by setting `webkitRelativePath` on dropped files via `Object.defineProperty`.

#### Component extraction ✅

- Extracted the upload UI into a reusable `FileDropZone.svelte` component:
  - **Props**: `class` (top-level styling), `disabled` (disables inputs/buttons), `accept` (file filter, default `image/*`), `files` (bindable `File[]`).
  - **Internal state**: `isDragOver`, `dragCounter`, file input refs, drag/drop handlers, directory traversal logic, file list rendering.
  - **Parent usage**: `<FileDropZone bind:files={selectedFiles} disabled={isSubmitting} />` — keeps `AddDatasetDialog` focused on orchestration (name input, upload action, progress, error display).

#### Client-side file type validation ✅

- Added `allowedExtensions` prop to `FileDropZone` (default matches backend `IMAGE_EXTENSIONS`): `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff`, `.tif`, `.ico`.
- Added `addFiles()` helper that filters incoming files:
  - Accepted files are appended to `files`.
  - Rejected files are tracked and shown in a warning banner (`alert-warning` style).
  - Warning shows up to 3 rejected filenames + "and N more" for brevity.
- Validation applies to both drag-drop and file picker inputs.
- Warning is cleared on the next file addition attempt.

#### Cancel upload ✅

- Added `signal?: AbortSignal` parameter to `uploadDataset()` and passed it through to `upload()`.
- In `AddDatasetDialog.svelte`:
  - Added `uploadAbortController` state.
  - Created `AbortController` at upload start, passed `.signal` to `uploadDataset()`.
  - Added `handleCancelUpload()` that calls `controller.abort()`.
  - Upload button replaced with **Cancel Upload** button (error-styled) while submitting.
  - `AbortError` is caught and shown as "Upload cancelled" instead of a generic error.
  - Controller is cleaned up in `finally`.

Phase 3 is complete.

### Phase 4: Sidecar file upload (.txt / .toml) ✅

- Added `UPLOAD_EXTENSIONS` constant in `datasets.py` = `IMAGE_EXTENSIONS | {".txt", ".toml"}`.
- Updated `create_dataset_from_upload()` to validate against `UPLOAD_EXTENSIONS` instead of `IMAGE_EXTENSIONS`.
- Updated `FileDropZone.svelte`:
  - Renamed `DEFAULT_IMAGE_EXTENSIONS` → `DEFAULT_UPLOAD_EXTENSIONS`, added `.txt` and `.toml`.
  - Added `children?: Snippet<[]>` prop — renders custom help text inside the drop zone when provided; falls back to default text otherwise.
  - `AddDatasetDialog` passes custom children: "Images, .txt captions, and .toml extras are supported."

### Phase 5: Separate root and folder storage ✅

- Refactored `create_dataset_from_upload()` in `datasets.py`:
  - Root files (no `/` in path) → written flat to `images/`.
  - Folder files (has `/` in path) → written to `folders/` preserving relative structure.
  - Config TOML has one `[[dataset]]` entry for `images/` and one per unique top-level folder in `folders/`.
  - Updated docstring to describe the new behavior.

### Phase 5b: Distinguish uploaded vs imported datasets ✅

Uploaded datasets (from the Upload tab) live entirely inside `STATE_PATH/<name>/` and are managed by yadc. Imported/created datasets (Import TOML and Create New tabs) reference external directories provided by the user.

**Backend:** `DatasetInfo.source` field already existed (`"upload" | "import" | "create"`), populated by `DatasetService.register()`.

**Frontend changes:**
- Added `source` to the `DatasetInfo` TypeScript interface in `datasetImages.ts`
- Dataset listing cards (`+page.svelte`): small "Managed" / "External" badge next to the dataset name
- Dataset detail topbar (`datasets/[name]/+page.svelte`): "Managed" / "External" badge next to the title
- Config tab (`DatasetConfig.svelte`): new read-only "Dataset" info section at the top
  - **Managed**: shows "Managed" badge + "Files stored in yadc state directory" + relative path (`<name>/config.toml` — state dir prefix stripped)
  - **External**: shows "External" badge + "References paths outside yadc" + full absolute `config_path`
- Props threaded through `SidePanel.svelte` (`source?: 'upload' | 'import' | 'create'`)

### Phase 6: Atomic upload failure handling ✅

Wrapped `create_dataset_from_upload()` in `try/except` that calls `shutil.rmtree(base_dir, ignore_errors=True)` on any exception, then re-raises. This cleans up partial uploads, `_register()` failures, etc.

Also added server-side accumulated size tracking during chunked file writes as a hard backstop for the `max_upload_size_bytes` limit. Replaced `shutil.copyfileobj()` with a manual read loop that tracks `total_written` and raises `ValueError` if exceeded.

### Phase 7: Code review fixes ✅

Fixed all must-fix and should-fix items from the code review:

1. **Partial upload cleanup** — `create_dataset_from_upload()` wrapped in try/except with `shutil.rmtree(base_dir)` on failure.
2. **`.` filename edge case** — Guard against empty, `.`, and `..` filenames before branch logic.
3. **Consistent path sanitization** — Root branch now uses `(images_dir / str(pure)).resolve()` instead of raw `filename`.
4. **Accumulated size check** — Hard backstop during chunked file writes.
5. **Extract `formatBytes()`** — Moved to `$lib/format.ts`, imported in `FileDropZone.svelte` and `AddDatasetDialog.svelte`.

### Phase 8: Full sidecar support (.history~ / .draft~) ✅

Backend `UPLOAD_EXTENSIONS` now includes `.txt`, `.toml`, `.toml~`, `.draft~`, `.history~`.

### Phase 9: Validate uploaded file contents ✅

Extension checks only verify the filename suffix. Added validation of the **actual content** of uploaded files:
- **Images**: `Image.open(stream).verify()` — skip corrupted/truncated images with warning.
- **`.toml` / `.toml~`**: `toml.loads(content.decode("utf-8"))` — skip invalid TOML with warning.
- **`.txt`, `.draft~`, `.history~`**: No content validation (text files).
- **Orphan sidecars**: After seeing all files, check that each sidecar has a matching image with the same stem in the same directory. Skip orphans with warning.

If all files are filtered out, the upload aborts with `ValueError("No valid files to upload after validation")` and the partial directory is cleaned up.

### Phase 10: Flatten uploads — no recursive folder scanning ✅

The dataset scanner (`_scan_dataset`) uses `img_dir.iterdir()` — it only reads **one level deep**. Uploads now enforce this flat structure:

**Backend:** `create_dataset_from_upload()` skips files with `len(pure.parts) > 2` (nested beyond `folder/file`).

**Frontend:**
- `collectFilesFromEntry()` (drag-and-drop): Only collects direct file children of dropped directories. Subdirectories are silently skipped.
- `handleFileSelect()` (`webkitdirectory`): Filters out files where `webkitRelativePath` contains more than one `/`.

### Phase 11: Upload response with warnings ✅

Created `DatasetUploadResult` dataclass (`dataset: DatasetInfo + warnings: list[str]`).
- Backend `create_dataset_from_upload()` returns `DatasetUploadResult`.
- API endpoint serializes it via `jsonify_dataclass()`.
- Frontend `uploadDataset()` returns `Promise<DatasetUploadResult>`.
- `AddDatasetDialog` shows `toast.warning()` with skipped file details when `warnings.length > 0`.

### Phase 12: Refactor AddDatasetDialog — split by tab, merge Import+Create ✅

**Split into self-contained tab components:**
- `UploadDatasetTab.svelte` — all upload state, FileDropZone, progress bar, cancel upload, toast warnings
- `CreateDatasetTab.svelte` — merged Import TOML + Create New with radio toggle
- `AddDatasetDialog.svelte` — thin shell: header + two tabs

**Merged Import + Create into one tab:**
- Radio toggle: "Add image paths" (default) vs "Import TOML config"
- Shared dataset name input across both modes
- Single submit button that calls `createDataset()` or `importDataset()` based on selection
- Button label adapts: "Create" / "Import" and "Creating…" / "Importing…"

### Phase 13: Stream validation progress to frontend ✅

The upload flow now streams server-side progress back to the client so there is no silent gap after the HTTP body is received.

**Implementation:** Chunked NDJSON streaming response via `XMLHttpRequest` — see `history/016-streaming-validation-progress.md` for the design analysis.

**Wire format:** NDJSON (`application/x-ndjson`), one JSON object per line, flushed after each event.

**Progress events (from server):**
- `{"phase": "validating", "file": "cat.jpg", "index": 5, "total": 100}` — during PIL verify / TOML parse
- `{"phase": "writing", "file": "cat.jpg", "index": 5, "total": 80}` — during file writes
- `{"phase": "complete", "dataset": {...}, "warnings": [...]}` — final result
- `{"phase": "error", "message": "..."}` — error result

**Files modified:**
- `yadc/api/services/dataset_upload.py` — `create_dataset_from_upload()` is now an async generator yielding `UploadProgressEvent` dataclasses
- `yadc/api/controllers/api_datasets.py` — upload endpoint returns a streaming `Response` with `application/x-ndjson` mimetype
- `yadc/webui/src/lib/upload.ts` — added `onChunk` callback to `UploadOptions`; parses `xhr.responseText` incrementally during download `onprogress`
- `yadc/webui/src/lib/stores/datasetImages.ts` — `uploadDataset()` passes `onChunk` through and resolves/rejects from `complete`/`error` events
- `yadc/webui/src/lib/components/datasets/UploadDatasetTab.svelte` — multi-phase progress bar (upload → validate → write)

### Phase 14 and beyond (pending)

Additional enhancements to be planned by user.

### Code review findings

Full implementation reviewed across all layers (backend, frontend, types). All linting and type checks pass cleanly. See `history/011-review.md` for the full record.

#### Must fix before merge

1. **Partial upload cleanup** — `create_dataset_from_upload()` creates directories before writing files; a mid-way failure or `_register()` failure leaves orphaned data. Wrap the entire method body in try/except that `shutil.rmtree(base_dir)` on any exception, then re-raises. This supersedes Phase 6 above with Option B.

2. **`.` filename edge case** — `PurePosixPath('.').parts == ('.',)` has `len == 1` → enters the root file branch. The `resolve()+relative_to()` check passes (directory is within itself). Then `open(dest_path, "wb")` raises `IsADirectoryError` → 500. Guard against `.` and empty filenames early.

3. **Root file path uses raw `filename` instead of `pure`** — In the root branch, `(images_dir / filename).resolve()` uses the raw user input, while the folder branch uses `str(pure)`. For consistency and clarity, both branches should use `str(pure)` (or `pure.name` for root files).

#### Should fix

4. **Size limit is best-effort only** — `request.content_length` is client-controlled and optional. A malicious client can bypass it by omitting the header. Add a server-side accumulated-size check inside the file write loop as a hard backstop.

5. **Duplicate `formatBytes()` function** — Identical implementation in `FileDropZone.svelte` and `AddDatasetDialog.svelte`. Extract to a shared utility (e.g., `$lib/format.ts`).

6. **Tab order vs. default mode mismatch** — Upload tab renders first in `PillTabs` but `mode` defaults to `'import'`, so the visual first tab doesn't match the selection. Either change the default to `'upload'` or reorder the tabs to put Import first.

#### Acknowledged / deferred

- **`UploadResponse` not exported** — Fine for now; `ResponseLike` interface is the public abstraction.
- **No loading state during drag-parsing** — Large folders may be slow to traverse; can add a spinner later if needed.
- **Phase 5b (distinguish uploaded vs imported)** — Still pending, tracked separately above.

### Post-implementation: Quart body-size limit

Quart's default `MAX_CONTENT_LENGTH` is **16 MB**. Uploads larger than that (e.g. a 140MB batch) hit Quart's built-in guard during `await request.form`, returning a generic "Request Entity Too Large" before our route handler even runs.

**Fix applied:**
- `application.py`: Set `app.config["MAX_CONTENT_LENGTH"] = configuration.max_upload_size_bytes` so Quart allows bodies up to our configured ~500MB limit.
- `api_datasets.py`: Moved `request.content_length` check to **before** `await request.form` so our `PAYLOAD_TOO_LARGE` error (with the actual byte limit in the message) is returned instead of Quart's generic one.

**Note on double-write:** Quart's multipart parser writes each file part to a `SpooledTemporaryFile` (on disk, not RAM) during parsing. Our code then reads from that temp file and writes to the final destination. So each uploaded byte hits disk twice: once as a temp file, once in `STATE_PATH/<name>/`. This is standard for multipart uploads. To avoid the double-write you'd need to bypass multipart entirely (e.g. raw per-file uploads with `request.stream`, or a protocol like `tus` for resumable streaming).

### Minor observations (no action needed)

See `history/013-review-minor-observations.md` for full details.

1. **`UploadResponse` class is not exported** — `upload.ts` defines `UploadResponse` as an unexported class. The `ResponseLike` interface in `api.ts` is the public abstraction for `apiErrorMessage()`, which is the right call. If a consumer ever needs to type-narrow or inspect `UploadResponse` directly, it would need to be exported, but there's no current need.

2. **`readDirectoryEntries` batching** — The `do...while` loop in `FileDropZone.svelte` correctly handles `readEntries()` not returning all entries in one call (some browsers return at most 100 entries per call). This is easy to get wrong and was done right.

3. **No loading state during drag-parsing** — Dropping a large folder triggers recursive `webkitGetAsEntry` traversal, which could be slow for deeply nested directories with thousands of files. Currently there's no spinner or "reading files…" indicator during traversal. Low priority — can be added later if users report issues.

## API Contract

### `POST /api/datasets/upload`

**Content-Type**: `multipart/form-data`

**Fields**:
- `name` (text, required): dataset name
- `files` (file(s), required): one or more files

**Allowed file types**: image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, `.bmp`, `.tiff`, `.tif`, `.ico`) + `.txt` caption sidecars + `.toml` extras sidecars

**Success (201)**: `DatasetInfo` JSON (same shape as other dataset endpoints)

**Errors**:
- `400` + `BAD_REQUEST`: missing name, no files, unsupported file extension, empty name, invalid file path
- `409` + `CONFLICT`: dataset name already exists
- `413` + `PAYLOAD_TOO_LARGE`: total upload size exceeds limit
