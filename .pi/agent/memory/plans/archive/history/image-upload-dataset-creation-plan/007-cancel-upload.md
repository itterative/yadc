# Phase 3: Cancel Upload

## Summary

Added upload cancellation support using `AbortController` and `AbortSignal`.

## Changes

### Modified Files

- `yadc/webui/src/lib/stores/datasetImages.ts`:
  - Added optional `signal?: AbortSignal` parameter to `uploadDataset()`.
  - Passed `signal` through to `upload()` utility.

- `yadc/webui/src/routes/AddDatasetDialog.svelte`:
  - Added `uploadAbortController: AbortController | null` state.
  - Added `handleCancelUpload()` function that calls `controller.abort()`.
  - Created `AbortController` at upload start and passed `.signal` to `uploadDataset()`.
  - Replaced the Upload button with a **Cancel Upload** button (error-styled border/background) while `isSubmitting`.
  - Caught `AbortError` (`DOMException` with `name === 'AbortError'`) in the catch block and shown "Upload cancelled" message.
  - Cleaned up `uploadAbortController` in `finally`.

## Verification

- `npx svelte-check --tsconfig ./tsconfig.json` → 0 errors, 0 warnings
- `uv run pytest tests` → 128 passed
