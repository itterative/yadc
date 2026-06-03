# Phase 3: Drag-and-Drop Zone

## Summary

Added a drag-and-drop zone to the Upload tab in `AddDatasetDialog`, allowing users to drop files and folders directly onto the dialog.

## Changes

### Modified Files

- `yadc/webui/src/routes/AddDatasetDialog.svelte`:
  - Added `isDragOver` reactive state and `dragCounter` counter for nested drag event handling.
  - Added `handleDragEnter`, `handleDragOver`, `handleDragLeave`, `handleDrop` event handlers.
  - Added helper functions for recursive directory traversal:
    - `readDirectoryEntries(reader)` — Promise wrapper for `FileSystemDirectoryReader.readEntries()`.
    - `getFileFromEntry(entry)` — Promise wrapper for `FileSystemFileEntry.file()`.
    - `collectFilesFromEntry(entry, path)` — recursively walks files and directories, preserving relative paths via `webkitRelativePath`.
  - Replaced the plain button group with a bordered drop zone:
    - Visual feedback on drag-over (accent border + tinted background).
    - "Drop files here" text changes when dragging over.
    - Buttons moved inside the drop zone for discoverability.

## Key Implementation Details

- **Nested drag events**: `dragCounter` increments on `dragenter` and decrements on `dragleave`. The drop zone highlight only clears when the counter reaches zero, preventing flicker when dragging over child elements.
- **Directory traversal**: `FileSystemDirectoryReader.readEntries()` may return partial results, so it's called in a loop until an empty batch is returned.
- **Path preservation**: Dropped files don't natively have `webkitRelativePath` set. The code uses `Object.defineProperty` to inject the reconstructed relative path, so `uploadDataset()` can send it to the backend via `FormData.append()`.
- **Tailwind opacity modifier**: `bg-accent/5` couldn't be used in a `class:` directive because `/` confuses the Svelte parser. Worked around by using a ternary expression in the `class` attribute instead.

## Verification

- `npx svelte-check --tsconfig ./tsconfig.json` → 0 errors, 0 warnings
- `uv run pytest tests` → 128 passed
