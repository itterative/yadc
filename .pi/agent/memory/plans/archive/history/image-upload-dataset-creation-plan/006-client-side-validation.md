# Phase 3: Client-Side File Type Validation

## Summary

Added client-side file extension validation to `FileDropZone.svelte` so unsupported files are rejected before being added to the selection.

## Changes

### Modified Files

- `yadc/webui/src/lib/components/ui/FileDropZone.svelte`:
  - Added `DEFAULT_IMAGE_EXTENSIONS` constant matching backend `IMAGE_EXTENSIONS`.
  - Added `allowedExtensions` prop (default: all image extensions). Empty array means "allow all".
  - Added `validationWarning` state for displaying rejected file messages.
  - Added `getExtension(filename)` helper — extracts lowercase extension from filename.
  - Added `isAllowedFile(file)` helper — checks extension against `allowedExtensions`.
  - Added `addFiles(newFiles)` helper:
    - Iterates through new files, splitting into accepted and rejected lists.
    - Appends accepted files to `files`.
    - Sets `validationWarning` with up to 3 rejected filenames + "and N more" suffix.
    - Clears previous warning on each new addition.
  - Updated `handleFileSelect()` to use `addFiles()` instead of direct assignment.
  - Updated `handleDrop()` to use `addFiles()` instead of direct assignment.
  - Added warning banner display below the drop zone using `alert-warning` style.

## Component API Update

```svelte
<FileDropZone
  bind:files={selectedFiles}
  allowedExtensions={['.jpg', '.png', '.txt', '.toml']}
/>
```

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `allowedExtensions` | `string[]` | Image extensions | File extensions to allow (lowercase, with dot). Empty array = allow all. |

## Verification

- `npx svelte-check --tsconfig ./tsconfig.json` → 0 errors, 0 warnings
- `uv run pytest tests` → 128 passed
