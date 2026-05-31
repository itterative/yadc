---
date: 2025-05-30
---
# Phase 1 Implementation: Dataset Entries + Extras Editing

**Context:** Users couldn't edit `[[dataset]]` entries (paths, extras/template variables) through the structured config form. Had to use raw TOML editor.

**Decision:** Added a "Paths & Variables" section in DatasetConfig with card-based entry editing and a reusable KeyValueEditor component.

**Implementation details:**
- `KeyValueEditor.svelte` — reusable component with `bind:entries` (array of `{key, value}`), add/remove rows, id prefix support
- `DatasetConfig.svelte` — new "Paths & Variables" section between Dataset info and API sections
  - Each `[[dataset]]` entry rendered as a card with: path input, variables (KeyValueEditor), inline image count badge, empty entry warning
  - Delete button on entries when more than one exists; "Add Path" button at bottom
  - `parsedDatasetRaw` state preserves original inline images across saves
  - `buildPatch()` includes full `dataset` array (atomically replaced via deep_merge)
  - `entriesEqual()` deep-comparison for dirty tracking
  - Preview `$effect` touches all entry fields for reactivity

**Key design choice:** Inline images are preserved by carrying `parsedDatasetRaw` (the original parsed entries) and copying their `images` arrays into the patch output. Inline image editing is deferred.

**Files touched:**
- NEW: `yadc/webui/src/lib/components/ui/KeyValueEditor.svelte` (86 lines)
- `yadc/webui/src/routes/datasets/[name]/DatasetConfig.svelte` (431→538, +107 lines)

**Checks:** eslint, prettier, svelte-check, build, pytest (168 passed) — all clean.
