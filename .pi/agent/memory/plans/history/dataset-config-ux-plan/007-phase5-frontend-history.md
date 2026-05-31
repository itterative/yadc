---
date: 2025-05-30
---
# Phase 5 Frontend: Config History UI

**Context:** Backend history API was ready (entry 006). Needed a UI to browse and restore config revisions.

**Implementation:**

### API Layer (`configs.ts`)
- `fetchConfigHistory` — debounced (same pattern as `fetchConfig`), with `_fetchConfigHistory` inner function
- `restoreConfigHistory` — POST to restore endpoint, returns updated config
- `ConfigHistoryEntry` type: `id`, `dataset_name`, `content`, `created_t`

### ConfigHistory.svelte (new component)
- Two-pane layout: narrow entry list (`w-40`, left) + TOML preview (right, read-only `TomlEditor`)
- Relative timestamps ("5m ago", "2h ago") with full date tooltips
- Restore button calls API, fires `onsaved`, refreshes list
- Empty state with `SvgFile` icon when no history exists
- Accepts `configVersion` prop — `$effect` tracks it alongside `datasetName` to auto-refresh on external saves
- Error display for failed requests

### SidePanel Integration
- Added `'history'` to `PanelTab` union type
- New History tab with `ConfigHistory` component
- `configVersion` counter in SidePanel — incremented by `handleConfigSaved()`, shared with both `DatasetConfig` and `ConfigHistory`
- Both components' `onsaved` → `handleConfigSaved()` → increments counter + calls parent `onconfigsaved`

### Reactivity Flow
1. User saves in Config tab → DatasetConfig fires `onsaved` → `handleConfigSaved()` → `configVersion++` + parent callback
2. ConfigHistory's `$effect` sees `configVersion` change → re-fetches history
3. User restores from History → same flow (restore also increments version)

**Files:**
- NEW: `yadc/webui/src/routes/datasets/[name]/ConfigHistory.svelte`
- Modified: `yadc/webui/src/lib/stores/configs.ts` (debounced history API)
- Modified: `yadc/webui/src/routes/datasets/[name]/SidePanel.svelte` (history tab + configVersion)

**Checks:** eslint, prettier, svelte-check, build, pytest (168 passed) — all clean.
