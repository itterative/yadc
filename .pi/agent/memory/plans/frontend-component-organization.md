---
name: frontend-component-organization
description: Restructure src/lib/components/ for consistency — merge dataset/+datasets/, split caption settings, group by domain. Established "promote on second use" and "feature folder" rules.
last_history: 001
---

# Frontend component organization

## History

- **001 (2026-06-03)** — Rebased onto `fix/svelte-frontend-misc-fixes`. Three frontend commits added `Card`/`ActionBar`/`ActionBarItem` primitives and grew `Caption.svelte`/`ImageDetail.svelte`/`DatasetManageTab.svelte`. Updated file sizes + target structure to include the new `ui/` primitives; demoted the `Caption.svelte` split from recommended to optional since the new Card/ActionBar structure already improves readability.

# Frontend component organization

## Goal

The frontend has grown organically into an inconsistent shape: `dataset/` vs `datasets/`, parent+children in two different arrangements, oversized files. Lock down conventions and reorganize for consistency without changing behavior.

## Conventions (locked-in)

1. **Where to put a component**:
   - `lib/components/ui/` — atomic UI primitives, no domain concepts, no API calls.
   - `lib/components/<domain>/` — domain components, used in ≥2 places OR used by a `lib/` component.
   - `routes/<path>/` (co-located) — used by exactly one route AND not by anything in `lib/`. Promote to `lib/` on second consumer.
2. **When to use a subfolder for a feature**:
   - ≥3 related `.svelte` files that form a coherent unit (host + tabs/sections), AND
   - sub-components used only by that host.
   - Folder name = feature name. Host file shares the folder name.
   - Inside the folder: `Feature.svelte`, `FeatureTab.svelte`, `state.svelte.ts` (if shared runes), `helpers.ts` (if pure helpers used by multiple files in the folder).
   - Pure helpers used by something **outside** the folder stay in `lib/` proper.

## Target structure

```
src/lib/components/
  ui/                              # atomic primitives — Dialog, Alert, Tabs, CodeMirror, Spinner, Tooltip, Toast, FileDropZone, Checkbox, KeyValueEditor, PromptPreview, Topbar, Card, ActionBar, ActionBarItem, etc.
  dataset/                         # MERGED from dataset/ + datasets/
    browser/                       # was: dataset/DatasetBrowser.svelte, dataset/DatasetImage.svelte
      DatasetBrowser.svelte
      DatasetImage.svelte
    detail/                        # was: dataset/ImageDetail/*  (renamed ImageDetail/ → detail/)
      ImageDetail.svelte           # 376 lines (after misc-fixes rebase) — host + data layer; could optionally be split into ImageDetail.svelte + ImageDetailTabs.svelte
      Caption.svelte               # 409 lines (after misc-fixes rebase) — caption box + drafts + history; uses Card/ActionBar/ActionBarItem so readability is fine; split now optional, not recommended
      Preview.svelte               # 14 lines (thin wrapper)
      Extras.svelte                # 80 lines (TOML editor)
    config/                        # was: datasets/DatasetConfig{Form,Advanced,History}.svelte + datasets/datasetConfig/
      DatasetConfig.svelte         # host
      DatasetConfigForm.svelte     # tab
      DatasetConfigAdvanced.svelte # tab
      ConfigHistory.svelte         # tab
      patch.ts
      state.svelte.ts
    upload/                        # was: datasets/DatasetUploadPanel.svelte, datasets/UploadDatasetTab.svelte, datasets/CreateDatasetTab.svelte
      DatasetUploadPanel.svelte    # main panel (create + append + conflict resolution)
      UploadDatasetTab.svelte      # thin wrapper for AddDatasetDialog
      CreateDatasetTab.svelte      # create/import form
    manage/                        # was: datasets/DatasetManageTab.svelte
      DatasetManageTab.svelte      # 183 lines (after misc-fixes rebase) — still single-file reasonable
  caption/                         # NEW — moved from routes/datasets/[name]/
    CaptionSettingsPanel.svelte    # was: routes/datasets/[name]/CaptionSettings.svelte (host)
    TemplateSection.svelte         # extracted from CaptionSettings
    OverridesSection.svelte        # extracted from CaptionSettings
  env/                             # NEW — extracted from dialogs/ and settings/
    EnvironmentSettings.svelte     # was: dialogs/EnvironmentSettings.svelte
    EnvSelector.svelte             # was: settings/EnvSelector.svelte
  export/                          # NEW — extracted from dialogs/
    ExportDialog.svelte            # was: dialogs/ExportDialog.svelte
  settings/                        # narrowed scope
    SettingsDialog.svelte          # was: dialogs/SettingsDialog.svelte
    GeneralSettings.svelte         # was: dialogs/GeneralSettings.svelte
    SecuritySettings.svelte        # was: dialogs/SecuritySettings.svelte
    CaptionOptionsFields.svelte    # was: settings/CaptionOptionsFields.svelte
  dialogs/                         # now only global utility dialogs
    PasswordPromptDialog.svelte    # the only remaining file (was: dialogs/PasswordPromptDialog.svelte)
  icons/                           # unchanged

src/routes/
  datasets/[name]/
    +page.svelte                   # unchanged structurally; just receives a slimmer CaptionSettingsPanel
    SidePanel.svelte               # unchanged
    AddFilesDialog.svelte          # unchanged
    DropUploadZone.svelte          # unchanged — still route-located (per "promote on second use" rule)
  AddDatasetDialog.svelte          # unchanged
  EditDatasetDialog.svelte         # unchanged
  EditTemplateDialog.svelte        # unchanged
  templates/+page.svelte           # unchanged
  +page.svelte                     # unchanged
  +layout.svelte                   # unchanged
```

### Sub-feature decision log

- **`dataset/upload/` (new subfolder)**: `DatasetUploadPanel`, `UploadDatasetTab`, `CreateDatasetTab` form a coherent unit. `UploadDatasetTab` and `CreateDatasetTab` are just slim wrappers around the panel for the two dialog entry points. All three files are dataset-upload-specific; worth grouping.
- **`dataset/manage/` (new subfolder)**: only one file today, but a sibling to `upload/` in the Add/Edit dialog (`EditDatasetDialog.svelte` switches between Config/Manage/Upload tabs). Grouped for symmetry — if it stays at 1 file it's still correct; if it grows, the folder is there.
- **`dataset/detail/` (rename)**: `ImageDetail/` → `dataset/detail/`. Sibling to `dataset/browser/`, `dataset/config/`, etc. — keeps the naming pattern consistent (one-word feature names).
- **`caption/` (new top-level domain)**: `CaptionSettingsPanel` is a domain component, not a route-specific file. The settings it exposes (env, options, template, overrides) are the **captioning** domain. Belongs at top level.
- **`env/` (new top-level domain)**: `EnvironmentSettings` (CRUD list + edit) and `EnvSelector` (caption-flow widget) are both env-related. Sharing a folder makes the relationship clear.
- **`export/` (new top-level domain)**: only one file today, but the export feature has a clear domain boundary (backend selection, source, format, output). Single-file domain folders are fine; this matches the "one-word feature name" pattern.
- **`settings/` (narrowed)**: now only the settings **domain** — `SettingsDialog` host + its tabs + the `CaptionOptionsFields` widget (which is a settings domain widget). No more env/ env selector in here.
- **`dialogs/` (narrowed)**: only `PasswordPromptDialog` — a truly-global utility dialog mounted in `+layout.svelte`. Consider promoting to `ui/` if it has no other peers in `dialogs/`.

## Phase 1: file moves (no behavior change)

Pure renames + folder reorganization. Each commit is a self-contained move. Mechanical, low risk.

### 1.1 Merge dataset/ + datasets/ into dataset/ subfolders

Steps:
1. Create `dataset/browser/`, `dataset/config/`, `dataset/upload/`, `dataset/manage/`.
2. Move:
   - `dataset/DatasetBrowser.svelte` → `dataset/browser/DatasetBrowser.svelte`
   - `dataset/DatasetImage.svelte` → `dataset/browser/DatasetImage.svelte`
   - `dataset/ImageDetail/*` → `dataset/detail/*` (rename folder too)
   - `datasets/DatasetConfig.svelte` → `dataset/config/DatasetConfig.svelte`
   - `datasets/DatasetConfigForm.svelte` → `dataset/config/DatasetConfigForm.svelte`
   - `datasets/DatasetConfigAdvanced.svelte` → `dataset/config/DatasetConfigAdvanced.svelte`
   - `datasets/ConfigHistory.svelte` → `dataset/config/ConfigHistory.svelte`
   - `datasets/datasetConfig/patch.ts` → `dataset/config/patch.ts`
   - `datasets/datasetConfig/state.svelte.ts` → `dataset/config/state.svelte.ts`
   - `datasets/DatasetUploadPanel.svelte` → `dataset/upload/DatasetUploadPanel.svelte`
   - `datasets/UploadDatasetTab.svelte` → `dataset/upload/UploadDatasetTab.svelte`
   - `datasets/CreateDatasetTab.svelte` → `dataset/upload/CreateDatasetTab.svelte`
   - `datasets/DatasetManageTab.svelte` → `dataset/manage/DatasetManageTab.svelte`
3. Delete the now-empty `datasets/` directory.
4. Update all `$lib/components/datasets/...` → `$lib/components/dataset/<sub>/<File>` imports across:
   - `routes/datasets/[name]/+page.svelte` (none directly, but check)
   - `routes/datasets/[name]/CaptionSettings.svelte` (CaptionOptionsFields + EnvSelector only)
   - `routes/datasets/[name]/AddFilesDialog.svelte` (DatasetUploadPanel)
   - `routes/datasets/[name]/SidePanel.svelte` (DatasetConfig, ImageDetail)
   - `routes/AddDatasetDialog.svelte` (UploadDatasetTab, CreateDatasetTab)
   - `routes/EditDatasetDialog.svelte` (DatasetConfig, DatasetManageTab, DatasetUploadPanel)
   - `routes/templates/+page.svelte` (no dataset imports)
   - `lib/components/dialogs/EnvironmentSettings.svelte` (no dataset imports)
   - any internal `lib/components/datasets/ConfigHistory.svelte`, etc.

### 1.2 Extract env/ and export/ domains

1. Create `env/`, `export/`.
2. Move:
   - `dialogs/EnvironmentSettings.svelte` → `env/EnvironmentSettings.svelte`
   - `settings/EnvSelector.svelte` → `env/EnvSelector.svelte`
   - `dialogs/ExportDialog.svelte` → `export/ExportDialog.svelte`
3. Update imports in:
   - `dialogs/SettingsDialog.svelte` (EnvironmentSettings)
   - `routes/datasets/[name]/CaptionSettings.svelte` (EnvSelector)

### 1.3 Move CaptionSettings to lib/components/caption/

This is the only Phase 1 step that involves moving **route files** to `lib/`. Defer the file-splitting (Phase 2) — for now just move the file as-is.

1. Create `caption/`.
2. Move `routes/datasets/[name]/CaptionSettings.svelte` → `lib/components/caption/CaptionSettingsPanel.svelte`.
3. Update the import in `routes/datasets/[name]/SidePanel.svelte`:
   - `import CaptionSettings from './CaptionSettings.svelte';` → `import CaptionSettings from '$lib/components/caption/CaptionSettingsPanel.svelte';`
4. Rename the in-page usage `<CaptionSettings ... />` → `<CaptionSettingsPanel ... />` (or keep the import alias as `CaptionSettings` if we want a smaller diff).

### 1.4 Narrow settings/ and dialogs/

1. Move `dialogs/SettingsDialog.svelte` → `settings/SettingsDialog.svelte`.
2. Move `dialogs/GeneralSettings.svelte` → `settings/GeneralSettings.svelte`.
3. Move `dialogs/SecuritySettings.svelte` → `settings/SecuritySettings.svelte`.
4. Move `settings/CaptionOptionsFields.svelte` → `settings/CaptionOptionsFields.svelte` (stays).
5. Update imports in `+layout.svelte` (SettingsDialog).
6. Update imports in `settings/SettingsDialog.svelte` (GeneralSettings, EnvironmentSettings, SecuritySettings — paths to `$lib/components/env/...` and `$lib/components/settings/...`).

### 1.5 Update memory

Update `.pi/agent/memory/frontend-architecture.md` to reflect the new structure. Add the "promote on second use" rule and the "feature folder" rule to the **Component Organization** section. Remove the stale `routes/datasets/[name]/DatasetConfig.svelte` reference.

Update `.pi/agent/memory/todo.md` — the "Sidebar / topbar refactor" entry can be folded into this work; remove the "Frontend remaining cleanup" entry that overlaps (SidePanel class prop, Tooltip z-index — those are separate issues).

## Phase 2: extract sub-components from oversized files

Behavior-preserving extractions. Bigger diffs but still mechanical.

### 2.1 Split CaptionSettingsPanel (683 → ~200 + 2 sub-components)

File after split:
- `lib/components/caption/CaptionSettingsPanel.svelte` (~200 lines) — host: state, env selector + options via existing `CaptionOptionsFields` + start/stop footer + override diff tracking. The "assembled options" / "persistSettings" / "handleStart" logic.
- `lib/components/caption/TemplateSection.svelte` (~180 lines) — extracted from the current `<section>` block: template list, selected template state, save/cancel/new logic, JinjaEditor with loading state.
- `lib/components/caption/OverridesSection.svelte` (~60 lines) — extracted from the current collapsible "Overrides" section. Receives `overrides`, `overridesExpanded` state, callbacks.

The `datasetDefaults` and `overrides` computation stay in the host (they integrate with `captionOptions` assembly). The `TemplateSection` receives `selectedTemplate`, `templateContent`, `templateDirty`, `isNewTemplate`, etc. as bindable props and emits a `dirty` change via callback.

### 2.2 Extract sub-components from `routes/datasets/[name]/+page.svelte` (631 → ~350)

- **`routes/datasets/[name]/DatasetTopbar.svelte`** (~100 lines, co-located) — the `<Topbar>` block: dataset name, source badge, captioning status row, refresh + upload buttons. Receives `datasetName`, `currentDataset`, `isBatchCaptioning`, `isStopping`, `captionPct`, `isRefreshing` as props + emit callbacks. The page keeps the `SetTopbar` wiring.
- **`lib/components/ui/EmptyState.svelte`** (small) — the "No images found" empty state. Generic title + description + optional action snippet. Reusable for the dataset listing page's empty state too.
- **Captioning-done effects** — extract into `lib/components/caption/useCaptioningEffects.svelte.ts` (a runes module exposing `useCaptioningDoneEffect(datasetName, onRefresh)`). Keeps the effect logic out of the page; the page just calls it.

After extraction, the page is mostly: state declarations + data-loading effects + layout. ~350 lines.

## Verification

After each phase:
- `cd yadc/webui && npx svelte-check --tsconfig ./tsconfig.json` — must pass.
- `npx eslint .` + `npx prettier --write .` — must pass.
- `npm run build` — must succeed.
- Run the webui dev server (tmux session) and click through: dataset listing → open dataset → side panel → caption settings → start captioning → image detail → caption history → drop-to-upload → edit config → env editor. Visual smoke test.

## Deferred (separate work)

- `lib/stores/datasetImages.ts` (656 lines) — split into `types.ts`, `api.ts`, `browser-store.ts`. Data-layer refactor, not a component-structure one. Should be its own plan. Tracked in `todo.md`.
- `lib/components/dialogs/EnvironmentSettings.svelte` (331 lines) — internally does env list + env editor + model picker. Could be split into `env/EnvironmentList.svelte` + `env/EnvironmentEditor.svelte`, but the current cohesion is OK. Revisit if it grows.
- Generic-ification of `DropUploadZone` vs `FileDropZone` — they share drop-handling logic. Defer until a clear duplication problem appears.
- `Caption.svelte` (409 lines after the misc-fixes rebase) — caption box + drafts + history. **Lower priority than before the rebase**: the file is now organized as `<Card>...<ActionBar>...</Card>` blocks using the new `Card`/`ActionBar`/`ActionBarItem` primitives, so the readability is much better. If we still want to split it, the natural split is into `CaptionBox.svelte` (display + edit + history list rendering) and `Drafts.svelte` (drafts list). But this is now optional, not recommended.
- `ImageDetail.svelte` (376 lines after the misc-fixes rebase) — could be split into `ImageDetail.svelte` (header + tab host) and `ImageDetailTabs.svelte` (the data layer + tab content), but the current cohesion is OK.
- `+page.svelte` (routes) — Phase 2.2 partially addresses. Full extraction (e.g. dropping the `createDatasetBrowserStore` factory since `+page.svelte` doesn't use it) is out of scope.
