# Code Review Fixes (Must + Should)

## Summary

Applied all must-fix and should-fix items from the code review in one pass.

## Changes

### Modified Files

- `yadc/api/services/datasets.py` — `create_dataset_from_upload()`:
  - Wrapped entire method body in `try/except` that calls `shutil.rmtree(base_dir, ignore_errors=True)` on any exception, then re-raises. Cleans up partial uploads, I/O failures, and `_register()` failures.
  - Added guard against empty, `.`, and `..` filenames before the root/folder branch logic.
  - Root branch now uses `(images_dir / str(pure)).resolve()` instead of raw `filename` for consistency with the folder branch.
  - Replaced `shutil.copyfileobj()` with a manual chunked read loop that tracks `total_written`. Raises `ValueError` if `total_written > max_upload_size_bytes`, acting as a hard server-side backstop for the size limit.

- `yadc/webui/src/lib/format.ts` — **New file**:
  - Extracted shared `formatBytes(bytes: number): string` utility.

- `yadc/webui/src/routes/AddDatasetDialog.svelte`:
  - Removed local `formatBytes()` function.
  - Added `import { formatBytes } from '$lib/format';`.

- `yadc/webui/src/lib/components/ui/FileDropZone.svelte`:
  - Removed local `formatBytes()` function.
  - Added `import { formatBytes } from '$lib/format';`.

## Verification

- `uv run ruff check yadc/api/services/datasets.py` → passed
- `uv run pytest tests` → 128 passed
- `npx svelte-check --tsconfig ./tsconfig.json` → 0 errors, 0 warnings
