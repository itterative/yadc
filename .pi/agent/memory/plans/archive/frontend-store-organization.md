---
name: frontend-store-organization
description: Restructure src/lib/stores/ for consistency — group by domain (dataset, caption, config, env, templates) into sub-folders, split datasetImages.ts into types/api. Top-level keeps global UI primitives (toasts, confirm, password, etc.) and the SSE event store.
last_history: 001
---

# Frontend store organization

## History

- **001 (2026-06-03)** — Phase 1 complete. 5 domain sub-folders created (`dataset/`, `caption/`, `config/`, `env/`, `templates/`). `datasetImages.ts` split into `types.ts` + `api.ts` + `index.ts`. Deprecated `captioning.ts` shim dropped. 36 import sites updated. 8 old files deleted. See `history/frontend-store-organization/001-phase-1-store-subfolders.md` for the full diff and deviations.

# Frontend store organization

## Goal

The frontend's `src/lib/stores/` has grown into a flat 16-file namespace (~2,233 lines) with no grouping by domain. Several files mix multiple concerns (types + API + actions + utility), and the `datasetImages.ts` file is 615 lines and was already deferred from the previous component-org plan for splitting. Lock down conventions and reorganize for consistency without changing behavior.

This plan mirrors the conventions established in `plans/archive/frontend-component-organization.md` — domain-grouped sub-folders for related concerns, single-purpose files at top level for global singletons, and re-export shims for back-compat.

## Conventions (locked-in)

1. **Where a store lives**:
   - `lib/stores/<domain>/` — stores serving a single domain (dataset, caption, config, env, templates), with a re-exporting `index.ts` so callers import from the domain root.
   - `lib/stores/` (top-level) — global UI primitives (toasts, confirm, password dialog, session password, settings, topbar) and the SSE event backbone. Singletons that don't form a coherent feature.
2. **When to use a sub-folder for a domain**:
   - ≥2 related files (types + API + store + actions) AND they all serve the same domain.
   - Folder name = the **domain** (bare noun, not the historical filename). Matches `lib/components/<domain>/` naming.
   - Inside the folder: `index.ts` (re-exports for back-compat), `types.ts` (data shapes), `api.ts` (transport — fetch/CRUD), plus `store.ts` / `actions.ts` when reactive state is involved, plus pure helpers (e.g. `jinja.ts`).
3. **Splitting an overgrown file**:
   - When a file mixes multiple concerns (types + API, or multiple unrelated APIs), split into one file per concern.
   - The split threshold is logical, not line-count — but a file > 500 lines is a strong signal.
4. **Imports**: `$lib/stores/<domain>` (re-exporting barrel) for cross-boundary refs, `./<file>` relative for files inside the same sub-folder. No file in `lib/` imports from `routes/` — preserved by the existing rule.
5. **Re-export shims**: each sub-folder has an `index.ts` re-exporting the public surface, so `$lib/stores/dataset` works for everything that previously imported from `$lib/stores/datasetImages`.

## Target structure

```
src/lib/stores/
  dataset/                        # NEW — was: datasetImages.ts (615 lines, deferred split from prior plan)
    index.ts                      # Re-exports types + API
    types.ts                      # All ~10 dataset-related types
    api.ts                        # All fetch/CRUD/upload/captioning helpers
  caption/                        # NEW — was: captionActions.ts + captionOptions.ts + captionSettings.ts (3 related files)
    index.ts
    actions.ts                    # captionOptions + lastStartedJobId + caption action functions
    options.ts                    # CaptionOptions type (was: captionOptions.ts)
    settings.ts                   # captionSettings storable (was: captionSettings.ts)
  config/                         # NEW — was: configs.ts (257 lines, mixed: types + config CRUD + export + drafts)
    index.ts
    types.ts                      # All config + export + draft types
    api.ts                        # Config CRUD + export + drafts API
  env/                            # NEW — was: envs.ts (153 lines, mixed: types + store + CRUD + key-mode)
    index.ts
    store.ts                      # envs reactive store + refreshEnvs
    api.ts                        # Env CRUD + model fetching + key-mode
  templates/                      # NEW — was: templates.ts (119 lines, mixed: types + store + CRUD + jinja helper)
    index.ts
    store.ts                      # templates reactive store + refreshTemplates
    api.ts                        # Template CRUD
    jinja.ts                      # extractVariables (pure helper)
  events.ts                       # UNCHANGED — single SSE event store (Zod schemas + types + stores + lifecycle)
  toasts.ts                       # UNCHANGED (top-level)
  confirm.ts                      # UNCHANGED (top-level)
  passwordPrompt.ts               # UNCHANGED (top-level)
  sessionPassword.ts              # UNCHANGED (top-level)
  storageStore.ts                 # UNCHANGED (top-level)
  settings.ts                     # UNCHANGED (top-level; UI settings + dialog state)
  topbar.svelte.ts                # UNCHANGED (top-level; rune module)
  captioning.ts                   # DELETED (was deprecated re-export shim, 0 callers)
```

### Per-domain decision log

- **dataset/** (sub-folder): `datasetImages.ts` (615 lines) was already deferred from the prior plan for splitting into `types.ts` + `api.ts`. Combining the split with the move to a sub-folder keeps the work cohesive. Even after splitting, `api.ts` will be ~540 lines — still large but every function is in scope ("fetch/CRUD/upload/captioning for the dataset domain"). Further splitting (e.g. `dataset/uploads.ts` separate from `dataset/captions.ts`) is possible but not planned.
- **caption/** (sub-folder): 3 related files (`captionActions.ts` 88 lines, `captionOptions.ts` 21 lines, `captionSettings.ts` 33 lines). The action functions, the type, and the persistent settings are all about captioning — naturally cluster. The `captionOptions.ts` is a type-only file but it's used by both `actions.ts` and `CaptionSettingsPanel.svelte`; keeping it in the folder makes the relationship clear.
- **config/** (sub-folder): `configs.ts` (257 lines) mixes 3 concerns: **config CRUD** (Pydantic-shaped), **export** (different domain — `lib/components/export/` is its own folder), and **drafts** (dataset-scoped but a config-adjacent concept). Per the user's "types + api" preference, we keep them all in `api.ts` for now and document that further splitting is a possible follow-up. The types file separates the data shape from the transport, which is the main win.
- **env/** (sub-folder): `envs.ts` (153 lines) mixes **reactive store** (`envs` + `refreshEnvs`) and **API helpers** (CRUD + model fetching + key-mode). Separating them clarifies what's a singleton state vs. a one-shot request. Key-mode is technically a security concern, but it lives on the same controller as the env API on the backend, so we keep it here.
- **templates/** (sub-folder): `templates.ts` (119 lines) mixes **reactive store**, **CRUD API**, and a **pure Jinja helper** (`extractVariables`). Separating the helper makes it independently testable and signals that it has no API/store dependencies. The `jinja.ts` name mirrors the backend `core/template.py`-style separation.
- **events.ts** (top-level, unchanged): 438 lines but a single coherent module — Zod schemas, types, internal stores, and SSE lifecycle are all part of "the SSE event store". Splitting would create artificial boundaries. If it grows past ~600 lines, revisit (e.g. `events/{types.ts, store.ts}`).

### What stays at top-level (and why)

The 7 files at top-level are heterogeneous singletons — no host+tabs pattern, no per-domain grouping, no natural feature boundary:

- `toasts.ts` — singleton notification state.
- `confirm.ts` — Promise-based dialog state.
- `passwordPrompt.ts` — Promise-based dialog state, but coupled to `sessionPassword.ts`.
- `sessionPassword.ts` — sessionStorage-backed value.
- `storageStore.ts` — generic factory (used by `sessionPassword.ts` + others).
- `settings.ts` — localStorage UI settings + dialog state.
- `topbar.svelte.ts` — rune module shared via `$state`.

Wrapping them in `lib/stores/ui/` would add a path segment without adding organization. The component `ui/` sub-folder is justified because its contents are atomic primitives that share a "no domain knowledge" trait; the store files don't have that self-similarity (a `storageStore` factory is not a primitive, a `topbar` rune module is not a dialog). Per the user's decision, they stay flat at top-level.

## Phase 1: per-domain sub-folders

Each step is a self-contained move: create the sub-folder with the new files, update all import sites via `sed`-style rewrites, delete the old files. Mechanical, low risk. Verification at the end of each step.

### 1.1 Move + split `datasetImages.ts` → `dataset/`

Steps:
1. Create `dataset/`.
2. Create `dataset/types.ts` with all ~10 types: `DatasetInfo`, `ImageInfo`, `ImagePage`, `DatasetUploadResult`, `UploadConflict`, `UploadProgressEvent`, `CaptioningJobInfo`, `CaptionData`, `HistoryEntry`, `DatasetFolder`, `DraftSummary`, `PromptPreview` (and `CaptioningJobInfo` lives here too).
3. Create `dataset/api.ts` with the same imports (`$lib/api`, `$lib/async`, `$lib/upload`, `./events` for `clientId`, `./sessionPassword` for `sessionPassword`) and the same exports (fetchDatasets, importDataset, createDataset, uploadDataset, appendUploadDataset, commitStagingUpload, deleteDatasetItems, fetchFolders, fetchDraftSummary, deleteDraftAll, deleteDataset, rescanDataset, fetchImages, fetchCaption, updateCaption, fetchHistory, restoreHistory, deleteHistory, deleteDraft, updateExtras, thumbnailUrl, fetchPromptPreview, mediaUrl, startCaptioning, fetchCaptioningStatus, stopCaptioning, captionSingleImage). No code changes — just move code.
4. Create `dataset/index.ts` re-exporting the public surface: `export * from './types'; export * from './api';`. This is what `$lib/stores/dataset` resolves to.
5. Update all 18 import sites: `$lib/stores/datasetImages` → `$lib/stores/dataset`.
6. Delete `datasetImages.ts`.
7. Verify: `npx svelte-check`, `npx eslint .`, `npm run build`.

Note: the prior `todo.md` entry for "Split `stores/datasetImages.ts` (~615 lines)" is satisfied by this step (types + api + index re-exports). The "browser-state.svelte.ts" follow-up is intentionally NOT part of this plan — it's a behavior refactor with a different risk profile (see Deferred section).

### 1.2 Move caption files → `caption/`

Steps:
1. Create `caption/`.
2. `git mv captionActions.ts caption/actions.ts` (and update its internal imports: `./datasetImages` → `../dataset`, `./events` stays as-is, `./passwordPrompt` stays, `./toasts` stays, `./captionOptions` → `./options`).
3. `git mv captionOptions.ts caption/options.ts`.
4. `git mv captionSettings.ts caption/settings.ts` (and update its import: `$lib/storable.js` stays — that path is unchanged).
5. Create `caption/index.ts` re-exporting: `export * from './actions'; export * from './options'; export * from './settings';`.
6. Update all import sites:
   - `$lib/stores/captionActions` → `$lib/stores/caption` (5 sites)
   - `$lib/stores/captionOptions` → `$lib/stores/caption` (1 site — `CaptionSettingsPanel.svelte`)
   - `$lib/stores/captionSettings` → `$lib/stores/caption` (1 site — `CaptionSettingsPanel.svelte`)
7. Delete the old `captionActions.ts`, `captionOptions.ts`, `captionSettings.ts` files.
8. Delete `captioning.ts` (deprecated re-export shim, 0 callers — confirmed via grep).
9. Verify.

### 1.3 Move config files → `config/`

Steps:
1. Create `config/`.
2. Create `config/types.ts` with all config-related types: `ExportBackend`, `ExportResult`, `DatasetConfig`, `Config`, `ConfigApi`, `ConfigPrompt`, `ConfigSettings`, `ConfigSettingsAdvanced`, `ConfigReasoning`, `ConfigReasoningAdvanced`, `ConfigDatasetEntry`, `ConfigValidationError`, `DatasetConfigDetail`, `ConfigHistoryEntry`.
3. Create `config/api.ts` with all API helpers (fetchExportBackends, runExport, fetchDatasetDrafts, fetchConfigs, fetchConfig, fetchConfigHistory, updateConfig, patchConfig, previewConfig, restoreConfigHistory, deleteConfig) and the same imports.
4. Create `config/index.ts` re-exporting: `export * from './types'; export * from './api';`.
5. Update all 7 import sites: `$lib/stores/configs` → `$lib/stores/config`.
6. Delete `configs.ts`.
7. Verify.

### 1.4 Move env files → `env/`

Steps:
1. Create `env/`.
2. Create `env/store.ts` with the `envs` writable + readonly, `refreshEnvs` action, and types `EnvInfo`, `EnvListResult`, `EnvStoreState`. Imports: `svelte/store`, `../async` (for `debounce` — but `fetchEnvs` is in api.ts now), and `./api` (for `fetchEnvs`).

   Actually, simpler: keep `fetchEnvs` in `api.ts` and import it in `store.ts`. The store module just owns the reactive state + the `refreshEnvs` action.
3. Create `env/api.ts` with all API helpers (`_fetchEnvs`, `fetchEnvs`, `saveEnv`, `deleteEnv`, `_fetchModels`, `fetchModels`, `revealEnvValue`, `fetchKeyMode`, `setKeyMode`, `changeKeyPassword`). Imports: `$lib/api`, `$lib/async`, `svelte/store` (for `get`), `./sessionPassword` (for `sessionPassword`).
4. Create `env/index.ts` re-exporting: `export * from './store'; export * from './api';`.
5. Update import sites: `$lib/stores/envs` → `$lib/stores/env` (2 sites — `events.ts` and `SecuritySettings.svelte`).
6. Delete `envs.ts`.
7. Verify.

### 1.5 Move templates files → `templates/`

Steps:
1. Create `templates/`.
2. Create `templates/store.ts` with the `templates` writable + readonly, `refreshTemplates` action, and types `TemplateInfo`, `TemplateListItem`, `TemplateStoreState`. Imports: `svelte/store`, `./api` (for `fetchTemplates`).
3. Create `templates/api.ts` with all API helpers (`_fetchTemplates`, `fetchTemplates`, `_fetchTemplate`, `fetchTemplate`, `saveTemplate`, `deleteTemplate`) and the regex constants `JINJA_VAR_RE`, `JINJA_FOR_RE`, `JINJA_BUILTINS`. Imports: `$lib/api`, `$lib/async`, `svelte/store`.
4. Create `templates/jinja.ts` with just `extractVariables()` and the regex constants it needs. Pure function, no other dependencies.
5. Create `templates/index.ts` re-exporting: `export * from './store'; export * from './api'; export * from './jinja';`.
6. Update import sites: `$lib/stores/templates` → `$lib/stores/templates` (no change! It's the same path. Only the file structure inside changes.) Wait — this is a special case. The import path is already `$lib/stores/templates` (matches the new sub-folder name), so **no import-site changes needed**. We only need to move code into the new sub-folder and delete the old flat `templates.ts`.

   Actually this is clean: the directory name happens to match the new sub-folder name. Callers' imports don't change. Just move the contents into the sub-folder.
7. Delete `templates.ts` (the old flat file).
8. Verify.

### 1.6 Update memory

Update `.pi/agent/memory/frontend-architecture.md`:
- Replace the `stores/` block in the directory structure tree with the new layout (5 sub-folders + 7 top-level files + 1 SSE file).
- Update the "Store Modules" table to reflect the new file paths. Each row's "File" column changes to the new location; "Purpose" stays the same.

Update `.pi/agent/memory/todo.md`:
- Mark the "Split `stores/datasetImages.ts`" entry as **DONE (Phase 1 of `plans/archive/frontend-store-organization.md`)**. Keep the "browser-state.svelte.ts" follow-up as a separate future item (it's a behavior refactor, not a structural one).

## Phase 2: optional cleanups (defer until needed)

- **`events.ts` split (438 lines)**: Single coherent module today. If it grows past ~600 lines, consider `events/{types.ts, store.ts}`. The natural split is the Zod schemas + types (`types.ts`, ~150 lines) vs. the internal stores + lifecycle (`store.ts`, ~290 lines). Not needed now.
- **`dataset/api.ts` further split (~540 lines)**: Could split into `dataset/uploads.ts` (create/append/commit/delete-items/folders/draft-summary, ~190 lines), `dataset/captions.ts` (caption/history/draft/extras/preview, ~140 lines), `dataset/captioning.ts` (start/stop/status, ~85 lines), and a slimmer `dataset/api.ts` (datasets list/CRUD + images fetch + URL builders, ~150 lines). The user opted for the simple types+api split; further splitting can be a follow-up.
- **`config/api.ts` further split (~120 lines)**: Could split export API into its own `config/export.ts` (since `lib/components/export/` is its own folder, paralleling it is consistent). And drafts API could move to `dataset/drafts.ts` (it's dataset-scoped, `/api/datasets/<name>/drafts`). The user opted for the simple types+api split; further splitting can be a follow-up.
- **`passwordPrompt.ts` and `sessionPassword.ts` grouped under `lib/stores/auth/`** (or similar): They're coupled (`withPasswordRetry` writes to `sessionPassword`), but they have different responsibilities (modal state vs. storage-backed value). Keep separate, no folder.

## Verification

After each step:
- `cd yadc/webui && npx svelte-check --tsconfig ./tsconfig.json` — must pass.
- `npx eslint .` + `npx prettier --write .` — must pass.
- `npm run build` — must succeed.
- Run the webui dev server (tmux session) and click through: dataset listing → open dataset → side panel → caption settings → start captioning → image detail → caption history → drop-to-upload → edit config → env editor → templates → export. Visual smoke test. In particular, verify:
  - Dataset page loads and SSE events still fire (captioning status updates).
  - Caption start works (validates `captionActions` + `dataset/api.ts` `startCaptioning`).
  - Image upload via drop works (validates `dataset/api.ts` upload helpers).
  - Env list refreshes on `environments_changed` event (validates `events.ts` → `env/store.ts` coupling).
  - Template list refreshes on `templates_changed` event (validates `events.ts` → `templates/store.ts` coupling).

## Deferred (separate work)

- **`dataset/browser-state.svelte.ts`**: Extract the page's inline `images` / `isLoading` / `loadInitial` / `loadMore` runes into a reusable state module. Behavior refactor with a different risk profile (changes how the page wires state). Keep as a follow-up; the structural move in 1.1 is independent.
- **`events.ts` split** (see Phase 2).
- **`config/api.ts` export/drafts split** (see Phase 2).
- **`dataset/api.ts` further split** (see Phase 2).
- **`passwordPrompt.ts` + `sessionPassword.ts` grouping** (see Phase 2).
