---
date: 2026-05-28
---
# Phase 1: Backend upload endpoint

**Context:** The image-upload-dataset-creation plan had no prior implementation. Phase 1 covers the backend API needed to receive uploaded image files and create a dataset from them.

**Decision:** Implemented `POST /api/datasets/upload` as a multipart form endpoint, `DatasetService.create_dataset_from_upload()`, and `Configuration.max_upload_size_bytes`.

**Rationale:**
- Quart's `await request.files` provides native async multipart handling, so no extra dependencies were needed.
- Uploaded files are stored under `STATE_PATH/<name>/data/` alongside the generated `config.toml` — this is self-contained and gets cleaned up automatically when the dataset is unregistered (existing `shutil.rmtree` behavior).
- Path sanitization (rejecting `..` and absolute paths) prevents directory traversal attacks from malicious `webkitdirectory` uploads.
- Relative directory structure is preserved so that folder uploads work as users expect.
- Size limit is checked against `request.content_length` (an upper-bound that avoids reading the entire stream just to check size).

**Files touched:**
- `yadc/api/configuration.py` — added `max_upload_size_bytes`
- `yadc/api/services/datasets.py` — added `create_dataset_from_upload()`
- `yadc/api/controllers/api_datasets.py` — added `upload_dataset()` route
