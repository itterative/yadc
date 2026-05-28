# Phase 4: Sidecar File Upload (.txt / .toml)

## Summary

Relaxed upload validation to allow `.txt` and `.toml` sidecar files alongside images, matching what the dataset scanner already supports.

## Changes

### Modified Files

- `yadc/api/services/datasets.py`:
  - Added `UPLOAD_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | frozenset({".txt", ".toml"})`.
  - Updated `create_dataset_from_upload()` to validate against `UPLOAD_EXTENSIONS` instead of `IMAGE_EXTENSIONS`.

- `yadc/webui/src/lib/components/ui/FileDropZone.svelte`:
  - Renamed `DEFAULT_IMAGE_EXTENSIONS` → `DEFAULT_UPLOAD_EXTENSIONS`.
  - Added `.txt` and `.toml` to the default allowed extensions list.
  - Updated help text: "Images, .txt captions, and .toml extras are supported."

## Verification

- `npx svelte-check --tsconfig ./tsconfig.json` → 0 errors, 0 warnings
- `uv run ruff check yadc/api/services/datasets.py` → passed
- `uv run pytest tests` → 128 passed
