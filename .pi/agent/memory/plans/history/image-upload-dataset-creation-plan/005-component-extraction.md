# Phase 3: Component Extraction

## Summary

Extracted the file upload UI (drop zone, file list, add/remove/clear) from `AddDatasetDialog.svelte` into a standalone `FileDropZone` component.

## Changes

### New Files

- `yadc/webui/src/lib/components/ui/FileDropZone.svelte` — Reusable file drop zone component.

### Modified Files

- `yadc/webui/src/routes/AddDatasetDialog.svelte`:
  - Removed all drag/drop logic, file input handling, directory traversal, file list rendering, and `formatBytes`.
  - Added `import FileDropZone from '$lib/components/ui/FileDropZone.svelte'`.
  - Replaced inline upload UI with `<FileDropZone bind:files={selectedFiles} disabled={isSubmitting} />`.
  - Kept dataset name input, upload button, progress bar, and error display in the parent.

## Component API

```svelte
<FileDropZone
  class="my-class"
  disabled={false}
  accept="image/*"
  bind:files={selectedFiles}
/>
```

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `class` | `string` | `""` | Additional CSS classes for the container |
| `disabled` | `boolean` | `false` | Disables file inputs and buttons |
| `accept` | `string` | `"image/*"` | File type filter for the file picker |
| `files` | `File[]` | `[]` | Bindable array of selected files |

## Key Implementation Details

- Uses Svelte 5 `$bindable()` for two-way `files` binding.
- The component is fully self-contained: drag/drop state, file input refs, directory traversal, file list rendering, and remove/clear actions all live inside it.
- The parent only needs to read `files` and handle the upload action.
- `formatBytes` was kept in the parent since it's also used for the upload progress display.

## Verification

- `npx svelte-check --tsconfig ./tsconfig.json` → 0 errors, 0 warnings
- `uv run pytest tests` → 128 passed
