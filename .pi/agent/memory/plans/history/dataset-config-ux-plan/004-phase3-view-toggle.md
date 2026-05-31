---
date: 2025-05-30
---
# Phase 3 Implementation: Simplified / Advanced View Toggle

**Context:** Users needed access to raw TOML editing inline in the Config tab, for fields not covered by the structured form (advanced settings, inline images, comments, etc.).

**Implementation:**
- Added `viewMode` state (`'simplified' | 'advanced'`) with segmented toggle ("Form" / "Raw TOML")
- Toggle sits at top of config content, before any sections
- **Simplified mode**: unchanged structured form with live TOML preview
- **Advanced mode**: full editable `TomlEditor` (reuses existing component with `editable={true}`)
- **Mode switching:**
  - Simplified → Advanced: transfers current `previewContent` (includes unsaved simplified changes) as raw editor content
  - Advanced → Simplified: switches view; if raw had unsaved changes, reloads from server to get clean structured state
- **Dirty tracking**: separate `simplifiedDirty` (structured fields) and `rawDirty` (raw content vs loaded snapshot), combined via `dirty` derived from active mode
- **Save routing**: PATCH in simplified mode, PUT (`updateConfig`) in advanced mode
- Extracted `populateFields()` from `loadConfig()` so advanced-mode save can re-populate structured fields from the PUT response

**Files touched:**
- `yadc/webui/src/routes/datasets/[name]/DatasetConfig.svelte` — added view mode state, toggle UI, advanced mode template, mode switching, dual save paths

**Checks:** eslint, prettier, svelte-check, build, pytest (168 passed) — all clean.
