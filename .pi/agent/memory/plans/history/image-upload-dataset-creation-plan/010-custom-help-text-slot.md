# Custom Help Text Slot in FileDropZone

## Summary

Added a `children` Snippet prop to `FileDropZone.svelte` so callers can override the default help text.

## Changes

### Modified Files

- `yadc/webui/src/lib/components/ui/FileDropZone.svelte`:
  - Added `children?: Snippet<[]>` prop.
  - Added `{#if children}` / `{:else}` block at the bottom of the drop zone:
    - If children provided: renders them in a `<div class="mt-2">`.
    - If not: shows default fallback text "Select individual images or an entire folder."

- `yadc/webui/src/routes/AddDatasetDialog.svelte`:
  - Updated `<FileDropZone>` usage to pass custom children:
    ```svelte
    <FileDropZone bind:files={selectedFiles} disabled={isSubmitting}>
      <p class="help-text">Images, .txt captions, and .toml extras are supported.</p>
    </FileDropZone>
    ```

## Usage

```svelte
<!-- Default help text -->
<FileDropZone bind:files={files} />

<!-- Custom help text -->
<FileDropZone bind:files={files}>
  <p class="help-text">Custom message here</p>
</FileDropZone>
```

## Verification

- `npx svelte-check --tsconfig ./tsconfig.json` → 0 errors, 0 warnings
