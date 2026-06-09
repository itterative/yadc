---
name: frontend/stores
description: Frontend lib/stores/ — 5 domain sub-folders (dataset, caption, config, env, templates) + 7 top-level singletons (events, toasts, confirm, password, etc.).
category: architecture
---

# Frontend: `lib/stores/`

5 domain sub-folders (each with a re-exporting `index.ts` so callers import from `$lib/stores/<domain>`) + 7 top-level singletons. The organization rules (when to introduce a sub-folder) are in `frontend-architecture`.

## See also (in `.pi/agent/memory/docs/`)

- `frontend-patterns` — SSE self-connecting module, Topbar pattern, Notifications
- `dataset-watcher` — Backend SSE event payload + suppression (referenced from SSE)

## Domain Sub-Folders

```
yadc/webui/src/lib/stores/
  dataset/                       # Dataset domain — types + API split
    index.ts                     # Re-exports for `$lib/stores/dataset`
    types.ts                     # DatasetInfo, ImageInfo, ImagePage, CaptionData, HistoryEntry, DatasetUploadResult, UploadConflict, UploadProgressEvent, CaptioningJobInfo, DatasetFolder, DraftSummary, PromptPreview
    api.ts                       # All fetch/CRUD/upload/captioning helpers (debounced via `debounce()`)
  caption/                       # Caption domain — actions + type + persisted settings
    index.ts                     # Re-exports for `$lib/stores/caption`
    actions.ts                   # captionOptions + lastStartedJobId + startBatchCaptioning/captionSingleImage/stopCaptioning (with toasts + optional onError callback)
    options.ts                   # CaptionOptions type (mirrors backend CaptionJobOptions)
    settings.ts                  # Last-used caption settings persisted to localStorage (env, maxTokens, imageQuality, etc.)
  config/                        # Config domain — types + API split
    index.ts                     # Re-exports for `$lib/stores/config`
    types.ts                     # Config + ConfigApi/ConfigPrompt/ConfigSettings/ConfigReasoning/ConfigDatasetEntry + DatasetConfig/Detail + ExportBackend/Result + ConfigHistoryEntry
    api.ts                       # Config CRUD + export API + drafts API. `fetchConfig`/`fetchConfigHistory` debounced.
  env/                           # Env domain — store + API split
    index.ts                     # Re-exports for `$lib/stores/env`
    store.ts                     # `envs` writable + `refreshEnvs` action + types
    api.ts                       # Env CRUD + model fetching + key-mode. `fetchEnvs`/`fetchModels` debounced.
  templates/                     # Templates domain — store + API + pure helper
    index.ts                     # Re-exports for `$lib/stores/templates`
    store.ts                     # `templates` writable + `refreshTemplates` action + types
    api.ts                       # Template CRUD. `fetchTemplates`/`fetchTemplate` debounced.
    jinja.ts                     # Pure `extractVariables()` (Jinja2 regex helpers)
```

## Top-Level Singletons

```
  events.ts                      # Self-connecting SSE store — opens TypedEventSource on load, pipes events into readonly writable stores (captioningStatus, pendingDatasetChanges, storedCaptions). Auto-refreshes envs/templates on change events. Caption text cached from SSE events (LRU).
  toasts.ts                      # Toast notification store + `toast.{info,success,warning,error}()` helpers
  confirm.ts                     # Promise-based confirmation dialog store
  passwordPrompt.ts              # Global password prompt store + `withPasswordRetry` + `PasswordPromptCancelled`
  sessionPassword.ts             # Tab-scoped in-memory password store
  storageStore.ts                # Generic `localStorage`/`sessionStorage`-backed writable factory
  settings.ts                    # UI settings (storable) + `settingsDialog` open-state
  topbar.svelte.ts               # Topbar `Snippet` shared via $state
```
