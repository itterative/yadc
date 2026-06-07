---
name: frontend/components-domain
description: Frontend lib/components/<domain>/ — dataset, caption, env, export, settings, dialogs, icons. Domain components (used in ≥2 places OR used by a lib/ component).
category: architecture
---

# Frontend: `lib/components/<domain>/`

Domain components. Per the "scope rule" (in `frontend-architecture`), these are components used in ≥2 places OR used by a `lib/` component. Each domain follows the "feature-folder rule" for sub-folders with ≥3 related files (host + tabs/sections).

## dataset/

```
yadc/webui/src/lib/components/dataset/
  browser/                      # Image grid + tile
    DatasetBrowser.svelte       # Masonry grid container (column distribution + infinite scroll + selectedId)
    DatasetImage.svelte         # Masonry grid tile (thumbnail + badges + selected outline)
  detail/                       # Image detail (tab host + tabs)
    ImageDetail.svelte          # Tab system host (CompactPillTabs: Caption / Preview / Edit) + data layer (captionData / historyEntries / API calls). History auto-loaded alongside caption.
    Caption.svelte              # Caption box (ActionCard + ActionBar) + Drafts (Card+ActionBar: Promote/Delete) + History (always visible when entries exist, Restore/Delete by content hash)
    RefineDialog.svelte         # Dialog for caption refinement — editable caption + feedback input, sends to model as conversation, shows result with Accept/Retry
    Preview.svelte              # Thin wrapper around PromptPreview
    Extras.svelte               # Always-editable TOML editor for the image's extras_raw (ActionCard + ActionBar with Save/Cancel)
  config/                       # Dataset config editor (tab host + tabs + helpers)
    DatasetConfig.svelte        # Tab system host (Form / Advanced / History) + data fetching + save orchestration
    DatasetConfigForm.svelte    # Structured form view of the config — fields bound to `configState` Svelte 5 `$state` rune from `state.svelte.ts`
    DatasetConfigAdvanced.svelte # Raw TOML editor view of the config. Binds to `configState.rawContent`; shows a warning when the structured form has unsaved changes.
    ConfigHistory.svelte        # Config revision history browser (list, view, restore entries). Uses ActionCard + ActionBar per entry. Reads from `ConfigHistoryRepository` on the backend.
    patch.ts                    # Pure TOML patch helpers (e.g. `assemblePatchFromForm`) — no Svelte/DOM dependencies, testable in isolation.
    state.svelte.ts             # Svelte 5 `$state` rune module holding the form / raw content state shared between `DatasetConfigForm` and `DatasetConfigAdvanced`. Exposes `simplifiedDirty` flag.
  upload/                       # Dataset upload (panel + slim tab wrappers)
    DatasetUploadPanel.svelte   # Shared upload panel (create + append modes). When `mode="append"`, accepts an `initialFiles` prop (consumed once via `untrack()`) to pre-populate the file list from a drop event on the page.
    UploadDatasetTab.svelte     # Upload tab — file selection, progress bar, cancel upload, toast warnings
    CreateDatasetTab.svelte     # Create/Import tab — radio toggle between "Add image paths" and "Import TOML config"
  manage/                       # Dataset manage (delete + folder ops)
    DatasetManageTab.svelte     # "Manage" tab inside the dataset creation dialog — list managed (upload-sourced) datasets with folder/image counts and delete actions.
```

## caption/

```
  caption/                        # Caption domain
    CaptionSettingsPanel.svelte   # The batch-captioning side panel. Derives isBatchCaptioning from store, shows Start/Stop button. Writes assembled options to captionActions store reactively. Fetches dataset config defaults, shows diff dots for overrides. Auto-saves overrides to `captionSettings` localStorage via `deferred()` on any field change; flushes pending save before starting captioning.
    OverridesSection.svelte       # Sub-section of the captioning panel: per-field reset/override UI for the localStorage-saved overrides
    TemplateSection.svelte        # Sub-section of the captioning panel: prompt template selection + variable editor
```

## env/

```
  env/                            # Env domain
    EnvironmentSettings.svelte    # Environment CRUD tab (env list, inline edit, token reveal)
    EnvSelector.svelte            # Environment form (env dropdown + URL/token/model, bindable props). "Manage…" link opens SettingsDialog at the Environments tab.
```

## export/

```
  export/                         # Export domain
    ExportDialog.svelte           # Export dialog (backend + draft/caption source selection)
```

## settings/

```
  settings/                       # Settings domain (dialog + tabs + shared widget + legacy)
    SettingsDialog.svelte         # App settings dialog (tab container)
    GeneralSettings.svelte        # General settings tab (browser notifications, thumbnails)
    SecuritySettings.svelte       # Security settings tab (key-mode switch, password change, YADC_PASSWORD warning)
    CaptionOptionsFields.svelte   # Shared widget — caption option fields (max tokens, image quality, rounds, reasoning, etc.) with optional diff-dot override indicators
    TemplateManager.svelte        # (LEGACY) Full template CRUD panel — superseded by dedicated /templates route
```

## dialogs/

```
  dialogs/                        # Global utility dialogs
    PasswordPromptDialog.svelte   # Global password prompt modal (driven by passwordPrompt.ts store)
```

## icons/

The icons live at `yadc/webui/src/lib/icons/` (sibling of `lib/components/`, not under it).

```
  icons/             # SVG icon components (Svg* prefix). Added manually from Material Symbols; standard viewBox `0 -960 960 960`. Sizing/color come from Tailwind classes via the `class` prop.
```

**Cross-references:**
- Drop-to-upload (DropUploadZone + AddFilesDialog): `frontend-patterns` → Drop-to-Upload
- Upload pipeline (backend side): `dataset-system` → Upload Pipeline
- Dataset creation flows: `dataset-system` → Three Creation Flows
- DatasetImage persistence (history, drafts): `dataset-system` → DatasetImage Persistence
