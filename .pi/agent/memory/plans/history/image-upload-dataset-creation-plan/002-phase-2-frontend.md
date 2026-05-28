# Phase 2: Frontend Upload UI

## Summary

Implemented the frontend upload flow for creating datasets from browser-selected files and folders.

## Changes

### New Files

- `yadc/webui/src/lib/upload.ts` — Promise-based `XMLHttpRequest` wrapper with progress tracking and `AbortSignal` support. Returns a fetch-like `UploadResponse` with `ok`, `status`, `text()`, `json()`.

### Modified Files

- `yadc/webui/src/lib/stores/datasetImages.ts`:
  - Imported `upload` and `UploadProgress` from `$lib/upload`.
  - Added `uploadDataset(name, files, onProgress?)` — builds `FormData` preserving `webkitRelativePath` for folder uploads, calls `upload()`, returns `DatasetInfo`.

- `yadc/webui/src/routes/AddDatasetDialog.svelte`:
  - Added `'upload'` to mode type.
  - Added upload state: `uploadName`, `selectedFiles`, `uploadProgress`, `fileInput`, `folderInput`.
  - Added `handleFileSelect`, `removeFile`, `clearFiles`, `formatBytes`, `handleUpload` functions.
  - Added "Upload" tab with:
    - Dataset name input
    - "Choose Files" / "Choose Folder" buttons triggering hidden inputs
    - Scrollable file list with names, sizes, per-file remove buttons
    - Total count/size summary with "Clear all"
    - Progress bar during upload
    - Upload button

- `yadc/webui/src/lib/api.ts`:
  - Added `ResponseLike` interface (`status`, `statusText`, `url`, `text()`).
  - Changed `apiErrorMessage(res: Response, ...)` → `apiErrorMessage(res: ResponseLike, ...)` so it accepts both native `Response` and `UploadResponse`.

- `yadc/api/controllers/utils_json.py`:
  - Added `PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"` to `ErrorCode` enum.

- `yadc/api/controllers/api_datasets.py`:
  - Changed size limit response to use `status=413, code=ErrorCode.PAYLOAD_TOO_LARGE`.

## Verification

- `npx svelte-check --tsconfig ./tsconfig.json` → 0 errors, 0 warnings
- `uv run ruff check yadc/api/...` → all passed
- `uv run pytest tests` → 128 passed
