---
name: dataset-config-ux-plan
description: Improvements to dataset config editing UX — structured form overhaul, dataset entries editing, simplified/advanced views, shared components, revision history, and TOML serialization.
status: In Progress
last_history: 12
category: meta
---

# Dataset Config UX Improvements Plan

## Overview

The dataset config editing experience needs improvement. The main gaps are:

1. **Dataset entries (`[[dataset]]`) are invisible** — users can't edit paths, extras (template variables like `style`, `artist`), or inline images through the structured form. Must use raw TOML.
2. **CaptionSettings and DatasetConfig duplicate fields** — both have max_tokens, image_quality, rounds, overwrite, reasoning settings with different patterns.
3. **No simplified/advanced toggle** — the Config tab shows a structured form + read-only TOML preview, but there's no way to switch to an editable raw TOML view inline.
4. **Missing advanced settings** — `system_role`, `user_role`, `assistant_role`, `assistant_prefill` aren't exposed in the structured form.
5. **No revision history** — PATCH/PUT writes directly to TOML with no backup. Mistakes are hard to undo.
6. **TOML serialization issues** — multiline strings and comment preservation (lower priority).

## Current Editing Surfaces

| Surface | Purpose | Persistence |
|---------|---------|-------------|
| **Config tab** (`DatasetConfig.svelte`) | Edit dataset TOML config (structured form) | TOML file (PATCH) |
| **Caption tab** (`CaptionSettings.svelte`) | Per-session caption settings | localStorage only |
| **EditDatasetDialog** | Raw TOML editor from listing page | TOML file (PUT) |

### Field Coverage Matrix

| Field | Config tab | Caption tab | EditDatasetDialog |
|-------|-----------|-------------|-------------------|
| API URL | ✅ | (via EnvSelector) | ✅ (raw) |
| Model name | ✅ | (via EnvSelector) | ✅ (raw) |
| Environment | ✅ | ✅ | ✅ (raw) |
| Template name | ✅ | ✅ | ✅ (raw) |
| Template content | ❌ | ✅ (JinjaEditor) | ✅ (raw) |
| Max tokens | ✅ | ✅ | ✅ (raw) |
| Image quality | ✅ | ✅ | ✅ (raw) |
| Rounds | ✅ | ✅ | ✅ (raw) |
| Overwrite | ✅ | ✅ | ✅ (raw) |
| Store conversation | ✅ | ❌ | ✅ (raw) |
| Reasoning | ✅ | ✅ | ✅ (raw) |
| Advanced settings | ❌ | ❌ | ✅ (raw) |
| **Dataset entries** | ❌ | ❌ | ✅ (raw) |
| **Extras (template vars)** | ❌ | ❌ | ✅ (raw) |
| Caption suffix | ❌ | ❌ | ✅ (raw) |

---

## Phase 1: Dataset Entries + Extras Editing

The highest-value gap. Users need to edit `[[dataset]]` entries — particularly `path` and `extras` (template variables).

### 1.1 Frontend: Dataset entries section in DatasetConfig

Add a new "Datasets" section above the existing "API" section in `DatasetConfig.svelte`.

**UI design:**
- Each `[[dataset]]` entry is a collapsible card showing:
  - **Path** — text input (the directory path)
  - **Extras** — key-value editor for template variables
    - Table/list of `key: value` pairs
    - "Add variable" button adds a new empty row
    - "×" button removes a row
    - Values are freeform strings (the most common type for extras)
  - **Inline images** — show count badge only ("3 inline images"), no editing (deferred to advanced/raw view)
  - If an entry has no path and no images, show a subtle warning ("Empty entry — set a path or add images")
- "Add dataset entry" button at the bottom to create a new `[[dataset]]` entry

**State management:**
- New `datasetEntries` state: array of `{ path: string, extras: Record<string, string>, imageCount: number }`
- Loaded from `config.parsed.dataset` on fetch
- Included in `buildPatch()` as the full `dataset` array (replacing it atomically via `deep_merge`)
- Dirty tracking: compare against `loadedDatasetEntries`

**Example TOML → form mapping:**

```toml
[[dataset]]
path = "/tmp/art_dataset"

[dataset.extras]
style = "watercolor"
artist = "unknown"
```

→ One card: path `/tmp/art_dataset`, extras: `style=watercolor`, `artist=unknown`.

### 1.2 Backend: No changes needed

`PATCH /configs/<name>` already deep-merges correctly — lists are replaced atomically. Sending `{"dataset": [...]}` in the patch body replaces the entire dataset array. The Pydantic `ConfigDatasetEntry` model already validates `path`, `images`, and `extras`.

### 1.3 Frontend: Key-value editor component

Create a reusable `<KeyValueEditor>` component for extras editing:

- **Props:** `bind:entries` (array of `{ key: string, value: string }`), `placeholder?`, `disabled?`
- **UI:** Simple list of key-value input pairs with add/remove buttons
- **Validation:** Empty keys are highlighted, duplicates warned
- Reusable for any dict-like TOML field

### Status

- [x] `<KeyValueEditor>` component
- [x] Dataset entries section in `DatasetConfig.svelte`
- [x] `buildPatch()` includes `dataset` array
- [x] Dirty tracking for dataset entries
- [x] Live preview updates for dataset entry changes

### Todos

- [ ] **Relative paths for uploaded datasets**: When `source === 'upload'`, display dataset entry paths relative to the dataset's state directory instead of showing the absolute state-folder paths. The backend's `DatasetInfo` likely has enough context; may need to pass the dataset root through to the component or compute relative paths on the backend.
- [ ] **Image/folder upload for managed datasets**: When `source === 'upload'`, allow uploading new images or folders to the dataset directly from the Paths & Variables section. Builds on the existing upload infrastructure (`DatasetUploadService`, `UploadDatasetTab`). Needs UX design — upload button per entry, or a shared upload action.
- [ ] **Path editing for external datasets**: Verify that changing a dataset entry's path works correctly with the backend's `DatasetWatcherService` (watchdog). Changing the path triggers a rescan, but the watcher may still be watching the old directory. Need to check if `rescan_dataset` re-registers watchers or if the watcher needs explicit path update handling.
- [x] **Extras value type support**: Extras values can be strings, numbers, booleans, or nested objects. Auto-detects types from parsed config, renders text input for strings, number input for numbers, checkbox for booleans, and read-only JSON preview for objects. Type selector dropdown (str/num/bool/obj) next to each key. Objects are not editable in the form — message prompts user to use Advanced view.

---

## Phase 2: ~~Advanced Settings Fields~~ Deferred to Advanced View

~Expose the `settings.advanced` and `reasoning.advanced` sub-fields in the structured form.~

These are niche power-user fields. Rather than adding them to the structured form, they'll be editable through the advanced/raw TOML view (Phase 3). No separate implementation needed.

---

## Phase 3: Simplified / Advanced View Toggle

Allow switching between the structured form and a raw TOML editor within the Config tab.

### 3.1 View mode toggle

Add a toggle in the Config tab header area (or in the PillTabs bar as a sub-toggle):

- **Simplified** (default): Current structured form layout. TOML preview is read-only at the bottom.
- **Advanced**: Full raw TOML editor (editable), like the current `EditDatasetDialog` but inline.

When switching from Advanced → Simplified, validate the TOML parses. If invalid, warn and stay in Advanced mode.

### 3.2 Advanced mode implementation

Reuse the `TomlEditor` component with `editable={true}`. Save via `PUT /configs/<name>` (full content replacement, same as `EditDatasetDialog`).

The structured form's `buildPatch()` → `PATCH` path continues to work in Simplified mode.

### 3.3 Consider: Replace EditDatasetDialog

Once the Config tab has an inline raw TOML editor, the `EditDatasetDialog` (accessible from the dataset listing page) becomes redundant. Options:

1. **Keep both** — EditDatasetDialog for quick edits from listing, Config tab for full editing (avoids navigating into the dataset page). Low cost.
2. **Remove EditDatasetDialog** — One less editing surface to maintain. Users must navigate to the dataset page → Config tab → Advanced mode.

Recommendation: Keep both for now. The listing-page dialog is convenient for quick edits without loading the full dataset browser.

### Status

- [x] View mode toggle UI (Form / Raw TOML segmented control)
- [x] Advanced mode: editable TomlEditor inline
- [x] Switch validation (Advanced → Simplified reloads from server if raw had unsaved changes)
- [x] Dirty tracking for raw TOML edits
- [x] Save via PUT in Advanced mode
- [x] `populateFields()` extracted for shared use between loadConfig and advanced save

---

## Phase 4: Shared Field Components

Reduce duplication between `CaptionSettings.svelte` and `DatasetConfig.svelte`.

### 4.1 Extract shared settings fields

Both components have nearly identical fields for: max_tokens, image_quality, rounds, overwrite, reasoning (enable + effort). Extract into a shared component:

**`<CaptionOptionsFields>`** component:
- Props: bindable values for each field, optional `diffDefaults` for showing diff dots
- Renders the Options + Reasoning sections
- Used by both `CaptionSettings.svelte` (with diff dots) and `DatasetConfig.svelte` (without diff dots)

### 4.2 CaptionSettings simplification

After extraction, `CaptionSettings.svelte` focuses on what's unique to it:
- EnvSelector (environment selection)
- Template section (JinjaEditor + template management)
- Overrides section
- Start/Stop button

The Options + Reasoning sections delegate to `<CaptionOptionsFields>`.

### Status

- [x] `<CaptionOptionsFields>` shared component
- [x] Integrate into `CaptionSettings.svelte`
- [x] Integrate into `DatasetConfig.svelte`
- [x] Verify diff dots still work in CaptionSettings

---

## Phase 5: Config Revision History

Track historical versions of the TOML config so users can revert mistakes.

### 5.1 Backend: SQLite config_history table

Add a new DB migration (step 5) to create a `config_history` table:

```sql
CREATE TABLE IF NOT EXISTS config_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL,
    content TEXT NOT NULL,
    created_t REAL NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_config_history_dataset ON config_history (dataset_name, created_t DESC);
```

On every `PATCH` or `PUT` to a dataset config, before writing the new content:

1. Read the current file content
2. Insert a row into `config_history` with `dataset_name`, current `content`, and timestamp
3. Then write the new content to disk

The `DatasetService` (or a new `ConfigHistoryService`) handles this via the existing `DBConnectionFactory`.

**New endpoints:**
- `GET /configs/<name>/history` — list history entries (id, timestamp, optional size/diff summary). Paginated or capped.
- `POST /configs/<name>/history/<id>/restore` — restore a specific history entry (overwrites current with the historical version, records the current version in history first)

**Pruning:** Keep a configurable max number of entries per dataset (default: 50). Prune on insert when exceeded.

**Advantages over sidecar file:** Works for all dataset sources (import/upload/create) without writing files next to external TOML configs. Consistent with the existing SQLite-based architecture. No file-format decisions.

### 5.2 Frontend: History browser

In the Config tab, the History is implemented as a third tab in `CompactPillTabs` (`Form` / `Advanced` / `History`).

**Target design — Incremental diff sections:**

Instead of a sidebar+preview split, each history entry is a section in a vertical stack. Sections are **always expanded** — no collapse/expand interaction. By default each section shows **only what changed in that save** (diff vs. the previous version). A toggle at the top of the section flips to the raw full snapshot.

```
┌─ Revision History ─────────────────────────────┐
│                                              ↻ │
├────────────────────────────────────────────────┤
│ ┌─ 2h ago — model_name changed ──────────────┐ │
│ │ May 30, 2026, 10:42 AM           [Restore] │ │
│ │                                [Show full] │ │
│ ├────────────────────────────────────────────┤ │
│ │ - model_name = "old-model"                 │ │
│ │ + model_name = "gemma3"                    │ │
│ └────────────────────────────────────────────┘ │
│ ─────────────────────────────────────────────  │
│ ┌─ 5h ago — added dataset entry ─────────────┐ │
│ │ May 30, 2026, 07:42 AM           [Restore] │ │
│ │                                [Show diff] │ │
│ ├────────────────────────────────────────────┤ │
│ │ + [[dataset]]                              │ │
│ │ + path = "/tmp/new_images"                 │ │
│ └────────────────────────────────────────────┘ │
│ ─────────────────────────────────────────────  │
│ ┌─ 1d ago — Initial config ──────────────────┐ │
│ │ May 29, 2026, 08:00 AM           [Restore] │ │
│ │                                [Show full] │ │
│ ├────────────────────────────────────────────┤ │
│ │ api.url = "http://localhost:11434"         │ │
│ │ model_name = "gemma3"                      │ │
│ │ ...                                        │ │
│ └────────────────────────────────────────────┘ │
│                                                │
│         [Show 5 older revisions]               │
└────────────────────────────────────────────────┘
```

**How diffs map to entries:**

History is saved pre-write, and the API returns newest-first. Each entry's default diff shows the changes introduced **by that save** (this version minus the one below it):

| Entry | Represents | Diff shows |
|-------|-----------|------------|
| A (top) | State after latest save | Changes made in latest save (A vs B) |
| B | State after previous save | Changes made in that save (B vs C) |
| C (bottom) | Initial config | Raw snapshot only (no previous) |

The current live config on disk is the same as the topmost history entry (both represent the state after the most recent save). No separate "Current" row is needed.

**Pagination:**

- Fetch 5 entries by default
- "Show N older" fetches the next batch via `before_id` cursor, appends to the list
- No page numbers — simple infinite-style loading

**Implementation notes:**

- Uses `@codemirror/merge` (`UnifiedMergeView`) for read-only diff rendering with green/red highlights and `+`/`-` gutter markers
- Each diff is a lightweight read-only CodeMirror instance — no accept/reject UI
- Restore button at the top of each section, styled as `.btn-secondary`, with `confirm()` dialog
- "Show full" / "Show diff" toggle at the top of each section — keeping buttons at the top ensures scroll position stays consistent when switching views
- Entry header shows relative time (primary) and absolute timestamp (secondary, muted)

**Known issues (moved to Phase 7):**
- Save/Reload footer visible and functional on History tab (7.3)

**Deferred:**
- Custom diff algorithm beyond `@codemirror/merge` (7.5)

### Status

- [x] DB migration: `config_history` table
- [x] Backend: insert history row on PATCH/PUT
- [x] Backend: `GET /configs/<name>/history` endpoint
- [x] Backend: `POST /configs/<name>/history/<id>/restore` endpoint
- [x] Backend: pruning (max entries per dataset)
- [x] Frontend: history section in Config tab
- [x] Frontend: preview + restore UI
- [x] `@codemirror/merge` dependency added, `UnifiedMergeView` wired into `TomlEditor`/`JinjaEditor`
- [x] `autoHeight` + `compactDiff` props added to editor components
- [x] Footer hidden on History tab; `activeView` type expanded to include `'history'`

---

## Phase 6: TOML Serialization Improvements

Lower priority. Two separate issues:

### 6.1 Multiline string serialization

Template values with newlines serialize as `\n` literals instead of `"""..."""` TOML multiline strings. This affects the `prompt.template` field and any extras that contain newlines.

**Approach:** Post-process the TOML output to detect string values containing `\n` and re-serialize them as `"""..."""`. Or use a TOML library that supports this natively.

**Investigation needed:**
- Does Python `toml` library have an option for multiline strings?
- Would switching to `tomlkit` (which preserves formatting/comments) solve both issues?

### 6.2 Comment preservation

Comments in TOML files are lost on parse → serialize because the TOML data model has no concept of comments.

**Approaches:**
- **tomlkit**: Python library that preserves comments and formatting. Would solve both multiline strings and comment preservation. Requires replacing the `toml` package.
- **Text-level patching**: For simple scalar changes, find/replace in the raw string instead of parse → serialize. Fragile for structural changes.
- **Accept the limitation**: Document that comments are not preserved when editing through the webui.

**Recommendation:** Evaluate `tomlkit` as a replacement for the `toml` package. If it works well, it solves both 6.1 and 6.2. If it has compatibility issues, accept the limitation for now and focus on text-level patching for scalar-only changes.

### Status

- [x] Add `tomlkit` dependency
- [x] Swap `api_configs.py` to use `tomlkit.loads`/`tomlkit.dumps`
- [x] Create `toml_merge()` in `dict_utils.py` — tomlkit-aware deep merge preserving comments
- [x] PATCH uses `toml_merge` (preserves comments on untouched sections)
- [x] PUT validates with `tomlkit.loads` (comments pass through verbatim)
- [x] All tests pass (168 passed)
- [ ] Post-completion: migrate CLI and other API services from `toml` to `tomlkit`

---

## Phase 7: Refinements

Post-implementation polish for issues discovered during use.

### Status

- [x] 7.1: Extras editing dirty tracking — `populateFields()` shared same array objects between `datasetEntries` and `loadedDatasetEntries`, so mutations were visible through both proxies. Fixed by deep-copying entries for the loaded snapshot.
- [x] 7.2: Snapshot strategy — changed from pre-write (old content) to post-write (new content). Added `_snapshot_initial()` for import/create/upload. Restore only saves current state before restoring (no redundant post-restore snapshot).
- [x] 7.3: Save/Reload footer visible on History tab — `activeView` expanded to `'simplified' | 'advanced' | 'history'`, footer conditionally hidden when on History.
- [x] 7.4: History tab layout — implemented incremental diff sections design. Stacked sections with inline diff + full-snapshot toggle eliminates the cramped sidebar+preview split.
- [x] 7.5: History diff view — `@codemirror/merge` (`UnifiedMergeView`) implemented for inline diff rendering. `collapseUnchanged` used with `minSize: 1` (not 0, which caused a RangeError).
- [ ] 7.6: **Form/Advanced view sync divergence** — When switching between Form and Advanced tabs, unsaved edits in one view are not reflected in the other. Current behavior: switching Form→Advanced syncs `previewContent` into the editor, but switching back with `rawDirty` reloads from server (discarding advanced changes). After the fix: both views preserve their own drafts, but they can diverge silently. Options: (1) Add inline sync notices ("Form has unsaved changes [Apply] [Keep]"), (2) Re-parse raw TOML into structured fields when switching Advanced→Form (instead of server reload), (3) Unify the two drafts into a single source of truth. **Deferred** — not actively blocking, but will confuse users who expect changes to propagate between tabs.

1. **Phase 1** (Dataset entries + extras) — highest user value
2. **Phase 2** (Advanced settings fields) — fills gaps in the structured form
3. **Phase 4** (Shared field components) — cleanup before Phase 3 to avoid duplicating the new fields
4. **Phase 3** (Simplified/Advanced toggle) — builds on a complete structured form
5. **Phase 5** (Revision history) — independent, can be done anytime after Phase 1
6. **Phase 6** (TOML serialization) — lowest priority, potential iceberg

---

## Key Files

| File | Role |
|------|------|
| `yadc/webui/src/routes/datasets/[name]/DatasetConfig.svelte` | Structured config editor |
| `yadc/webui/src/routes/datasets/[name]/CaptionSettings.svelte` | Per-session caption settings |
| `yadc/webui/src/routes/datasets/[name]/SidePanel.svelte` | Tab container |
| `yadc/webui/src/routes/EditDatasetDialog.svelte` | Raw TOML editor dialog |
| `yadc/webui/src/lib/stores/configs.ts` | Config API helpers + types |
| `yadc/api/controllers/api_configs.py` | Config CRUD endpoints |
| `yadc/core/config.py` | Pydantic config models |
| `yadc/utils/dict_utils.py` | `deep_merge()` |

## Open Questions

1. **Inline images editing** — Should the structured form support editing inline `[[dataset.images]]`? Deferred to the advanced/raw TOML view for now, but worth considering for a future iteration.
2. **KeyValueEditor value types** — Extras can be strings, numbers, or booleans. Should the editor auto-detect types or always use string inputs? Recommend: always strings for simplicity, with a note that non-string values require the raw TOML editor.
3. **CaptionSettings vs DatasetConfig unification** — Should we eventually merge these into a single panel? The current separation (per-session vs persistent) is clear, but having two panels with overlapping fields is confusing. Recommend: keep separate for now but use shared components (Phase 4).
4. ~~**History file location**~~ — Resolved: using SQLite `config_history` table instead of sidecar files. Works for all dataset sources without writing files next to external TOML configs.
