---
name: frontend/stores
description: Frontend lib/stores/ — 5 domain sub-folders (dataset, caption, config, env, templates) + 7 top-level singletons (events, toasts, confirm, password, etc.).
category: architecture
---

# Frontend: `lib/stores/`

5 domain sub-folders (each with a re-exporting `index.ts` so callers import from `$lib/stores/<domain>`) + 7 top-level singletons. The organization rules (when to introduce a sub-folder) are in `frontend-architecture`.

## See also (in `.pi/agent/memory/docs/`)

- `frontend-patterns` — Abort contexts, SSE self-connecting module, Topbar pattern, Notifications
- `dataset-watcher` — Backend SSE event payload + suppression (referenced from SSE)

## Convention: events.ts is the router, domain stores own the state

`lib/stores/events.ts` is **only** the SSE transport + event-routing layer. It defines the Zod schemas and wires each event to a writer in the appropriate domain store. The writable stores themselves live in the domain sub-folder (e.g. `caption/status.ts` owns `captioningStatuses` and its `setCaptioningStatus` / `resetCaptioningStatus(datasetName)` actions). This keeps `events.ts` small and means each store can be reasoned about (and tested) in isolation.

## Domain Sub-Folders

```
yadc/webui/src/lib/stores/
  dataset/                       # Dataset domain — types + API split
    index.ts                     # Re-exports for `$lib/stores/dataset`
    types.ts                     # DatasetInfo, ImageInfo, ImagePage, CaptionData, HistoryEntry, DatasetUploadResult, UploadConflict, UploadProgressEvent, CaptioningJobInfo, DatasetFolder, DraftSummary, PromptPreview
    api.ts                       # All fetch/CRUD/upload/captioning helpers. Accepts optional `signal?: AbortSignal` as the last positional parameter. Debounced helpers (`fetchDatasets`, `fetchImages`, `fetchCaption`, `fetchHistory`, `fetchPromptPreview`) use the `lib/async.ts` AbortSignal convention.
    pending.ts                   # pendingDatasetChanges store + addPendingDatasetChange/clearPendingDatasetChange. Seeded by the SSE `dataset_changed` handler.
  caption/                       # Caption domain — actions + many focused concern files
    index.ts                     # Re-exports for `$lib/stores/caption`
    actions.ts                   # captionOptions + lastStartedJobId + startBatchCaptioning/captionSingleImage/refineCaption/stopCaptioning (with toasts + optional onError callback)
    options.ts                   # CaptionOptions type (mirrors backend CaptionJobOptions)
    settings.ts                  # Last-used caption settings persisted to localStorage (env, maxTokens, imageQuality, etc.)
    status.ts                    # captioningStatuses store (per-dataset Map keyed by dataset_name) + setCaptioningStatus (upsert + schedule terminal eviction) + resetCaptioningStatus(datasetName, force?). Gated reset preserves 'running'/'stopping' state. Terminal entries auto-evict after 5 min so several concurrent jobs' results stay visible on the dataset list. INITIAL_CAPTIONING_STATUS constant.
    inflight.ts                  # currentlyCaptioning derived (set view) + _currentlyCaptioningMap internal + addCurrentlyCaptioning/removeCurrentlyCaptioning/clearCurrentlyCaptioning
    jobs.ts                      # _activeJobIds (bounded ring) + registerJobId + isOwnJobId. Used by the SSE `dataset_changed` handler to suppress events caused by our own jobs.
    refined.ts                   # imageRefined store + setImageRefined + consumeImageRefined (multi-key match by imageId/source/draftName)
    timing.ts                    # captionTimingRing (per-API+model, persisted to localStorage, cap=32 samples) + recordCaptionTiming
  config/                        # Config domain — types + API split
    index.ts                     # Re-exports for `$lib/stores/config`
    types.ts                     # Config + ConfigApi/ConfigPrompt/ConfigSettings/ConfigReasoning/ConfigDatasetEntry + DatasetConfig/Detail + ExportBackend/Result + ConfigHistoryEntry
    api.ts                       # Config CRUD + export API + drafts API. Accepts optional `signal?: AbortSignal` as the last positional parameter. `fetchConfig`/`fetchConfigHistory` debounced.
  env/                           # Env domain — store + API split
    index.ts                     # Re-exports for `$lib/stores/env`
    store.ts                     # `envs` writable + `refreshEnvs` action + types. `refreshEnvs(signal?)` skips updating the store if the signal was aborted.
    api.ts                       # Env CRUD + model fetching + key-mode. Accepts optional `signal?: AbortSignal` as the last positional parameter. `fetchEnvs`/`fetchModels` debounced.
  templates/                     # Templates domain — store + API + pure helper
    index.ts                     # Re-exports for `$lib/stores/templates`
    store.ts                     # `templates` writable + `refreshTemplates` action + types. `refreshTemplates(signal?)` skips updating the store if the signal was aborted.
    api.ts                       # Template CRUD. Accepts optional `signal?: AbortSignal` as the last positional parameter. `fetchTemplates`/`fetchTemplate` debounced. `duplicateTemplate` is a frontend-only helper (GET source + PUT new name) — no backend endpoint.
    jinja.ts                     # Pure `extractVariables()` (Jinja2 regex helpers)
  prompts/                       # Prompt-generator domain — streaming generation
    index.ts                     # Re-exports for `$lib/stores/prompts`
    types.ts                     # ExamplePair + PromptGenFocus + PromptGenRequest + StreamEvent + GenerationStatus + PromptGenMode + PromptHistoryListItem (summary row for the history list — no full examples, `intent_preview` truncated server-side, `had_template` derived server-side from `template_content IS NOT NULL`, `example_count` derived from a SQL `LEFT JOIN ... GROUP BY` on the sibling `prompt_example_images` BLOB table) + PromptHistoryEntry (full entry for restore — includes full `intent`, `examples` array, `template_content: string | null`) + SaveHistoryArgs (body for `POST /api/prompts/history` — snake_case `template_content` to match the wire format).
    api.ts                       # `generatePrompt()` — POST to /api/prompts/generate, iterate NDJSON stream via ReadableStream reader, dispatch token/reasoning/done/error callbacks. `fetchImageAsDataUrl()` (browser fetch + FileReader) and `readFileAsDataUrl()` (local File → data URL) for the multimodal examples panel. History endpoints: `fetchHistoryList(signal?)` → `PromptHistoryListItem[]` (returns `data.entries ?? []` so missing field is tolerated), `fetchHistoryEntry(id, signal?)` → `PromptHistoryEntry` (full entry, for restore), `saveHistoryEntry(args, signal?)` → `PromptHistoryEntry` (POSTs the body, returns the saved entry), `deleteHistoryEntry(id, signal?)` → `void` (DELETE the entry, no return).
    store.svelte.ts              # Svelte 5 `$state` rune module — generation status (idle/streaming/done/cancelled/error), accumulated body/reasoning, active AbortController. State mutators (beginGeneration, appendToken, completeGeneration, cancelGenerationState, failGeneration, reset) own the AbortController lifecycle so cancel can abort it.
    actions.ts                   # `startGeneration()` — wires the streaming callbacks to the state mutators and surfaces errors via toasts + the generation state. `StartGenerationArgs` adds `templateContent`, `maxTokens`, and `imageQuality` (pass-through to `template_content` / `max_tokens` / `image_quality` on the request); `saveHistoryEntry` is imported from `./api` and re-exported here as a convenience (not an action — the host calls it directly in the Save-to-history button handler).
    settings.ts                  # Persisted form settings (mode, env, apiUrl, apiToken, apiModelName, intent, focus, maxTokens, imageQuality) backed by localStorage via the shared `storable` utility. `promptSettings` is a Svelte writable store; `resetPromptSettings()` resets to defaults (also fires through the store, so it gets persisted). `examples` are NOT persisted (image data URLs are too large; no re-fetchable source per example yet — superseded by the server-side prompt-history persistence, which lets the user save/restore the examples via the History tab). `templateContent` and the template picker selection are NOT persisted (ephemeral local state — the user re-selects a template to re-fetch on reload). Storage key: `yadc/prompts/formSettings` (`$version: 3`; a single shape-agnostic `migrateToCurrent` overlays the current defaults onto any prior stored object so missing fields — `mode` (added v2), `maxTokens` / `imageQuality` (added v3) — fall back to defaults while the user's existing settings are preserved). `mode` is persisted so the user comes back to the same New/Refine selection (the form's inline mode toggle). Follows the same shape as `stores/settings.ts` and `stores/caption/settings.ts`. The host component uses the controlled-component pattern (local `$state` initialized from `$promptSettings`, single `$effect` syncing back via `promptSettings.update`) — `bind:prop={$store.field}` does NOT work reliably in Svelte 5 (the `$store` auto-subscription is read-only). First user of the `storable` `migrate` argument pattern (see `storable.js`); the bug fix in `storable.js` that actually writes the migrated data back to the store was necessary for the migration to take effect.
    api.test.ts                  # Vitest unit tests for the NDJSON streaming reader (happy path, chunked streaming, errors, abort, FileReader helper).
```

**Naming convention for sub-folder files**: most sub-folders use a role-based convention (`types.ts`, `api.ts`, `store.ts`, `actions.ts`). When a single domain has many concerns (e.g. `caption/`), it can also use feature-based naming — one file per concern (`status.ts`, `inflight.ts`, `timing.ts`, etc.), where each file owns the store, the actions, and the types for that one concern. This scales better than the role-based convention when a domain grows past 3-4 files.

## Top-Level Singletons

```
  events.ts                      # Self-connecting SSE event router. Opens TypedEventSource on load, validates events with Zod, and routes each event to a writer in the appropriate domain store. Owns the connection lifecycle (auto-reconnect, mobile visibility recovery, fallback interval, resumption-failed flag) + `clientId` (tab identity for suppression) + Zod schemas + the pure event mirrors that don't belong in any domain (resumptionFailed, lastCaptionedImage, lastCaptionError). Does NOT own domain state.
  toasts.ts                      # Toast notification store + `toast.{info,success,warning,error}()` helpers
  confirm.ts                     # Promise-based confirmation dialog store
  passwordPrompt.ts              # Global password prompt store + `withPasswordRetry` + `PasswordPromptCancelled`
  sessionPassword.ts             # Tab-scoped in-memory password store
  storageStore.ts                # Generic `localStorage`/`sessionStorage`-backed writable factory
  settings.ts                    # UI settings (storable) + `settingsDialog` open-state
  topbar.svelte.ts               # Topbar `Snippet` shared via $state
```
