# Refactor AddDatasetDialog — Split by Tab, Merge Import+Create

## Summary

Refactored the monolithic `AddDatasetDialog.svelte` into self-contained tab components and merged the Import TOML and Create New tabs into one.

## Changes

### New Files

- `yadc/webui/src/lib/components/datasets/UploadDatasetTab.svelte` — Self-contained upload tab:
  - All upload state (name, files, progress, abort controller)
  - FileDropZone with allowed extensions
  - Info box about one-level nesting and validation
  - Progress bar + cancel button
  - Error display + toast warnings on skip
  - Calls `uploadDataset()` and emits `oncreated` / `onclose`

- `yadc/webui/src/lib/components/datasets/CreateDatasetTab.svelte` — Merged create/import tab:
  - Radio toggle: "Add image paths" (default) vs "Import TOML config"
  - Shared dataset name input
  - Conditional form: textarea for paths (Create mode) or text input for TOML path (Import mode)
  - Single submit button adapting label + loading text based on mode
  - Calls `createDataset()` or `importDataset()` accordingly
  - Error display

### Modified Files

- `yadc/webui/src/routes/AddDatasetDialog.svelte` — Slim shell:
  - Removed all tab-specific state and handlers
  - Just header + `PillTabs` with two tabs
  - Each tab renders its self-contained component passing `oncreated` and `onclose`

## Verification

- `npx svelte-check --tsconfig ./tsconfig.json` → 0 errors, 0 warnings
- `uv run pytest tests` → 128 passed
