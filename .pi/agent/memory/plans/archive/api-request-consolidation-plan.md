---
name: api-request-consolidation-plan
description: Consolidate redundant API requests — env list/details, dataset browser, SSE caption text.
status: Complete
last_history: 13
category: meta
---

# API Request Consolidation

## Problem Statement

The webui makes more API requests than necessary in several common workflows:

1. **Environment list vs details**: `GET /api/envs` returns only names, so `EnvSelector` must call `GET /api/envs/{name}` individually after the user selects one.
2. **Dataset browser loads 4 requests**: Loading a dataset page triggers `GET /images`, `GET /datasets`, `GET /caption`, and `GET /configs/{name}` in parallel.
3. **Environment call repeats**: Opening CaptionSettings on a dataset page re-triggers the environment fetch (same as #1).
4. **SSE caption text missing**: `ImageCaptionedEvent` sends image metadata but not the caption text, so selecting a captioned image still requires `GET /images/{id}/caption`.

## Phase 1: Environment Details Endpoint ✅ Implemented (Option B)

**Decision:** Changed `GET /api/envs` to return `list[EnvInfo]` directly instead of adding a separate `/details` endpoint. The payload size is negligible (users typically have <10 envs) and the API surface is simpler.

**Backend:**
- Changed `GET /api/envs` → returns `list[EnvInfo]` (all envs with full details in one call)
- Extracted `_format_env()` helper shared with `GET /envs/<name>`

**Frontend:**
- Changed `EnvStoreState.items` from `string[]` to `EnvInfo[]`
- Updated `fetchEnvs()` and `refreshEnvs()` to return `EnvInfo[]`
- Updated `EnvSelector.svelte` to derive `envInfo` from `$envs.items` directly — no per-selection `fetchEnv()` call
- Updated `EnvironmentSettings.svelte` to remove `envDetailMap` and the N per-env `fetchEnv()` calls it made on load. `startEditEnv()` now receives the `EnvInfo` object directly.

**Files:** `yadc/api/controllers/api_envs.py`, `yadc/webui/src/lib/stores/envs.ts`, `yadc/webui/src/lib/components/settings/EnvSelector.svelte`, `yadc/webui/src/lib/components/dialogs/EnvironmentSettings.svelte`

## Phase 1.5: SSE Environment & Template Change Events ✅ Implemented

**Goal:** Push environment and template list updates to all connected clients via SSE so frontends stay in sync without manual refreshes.

**Decision:** Use filesystem watchers (`watchdog`) instead of dispatching from API controllers. This catches both webui edits and CLI changes uniformly.

### Environments

**Backend:**
- Added `EnvironmentsChangedEvent` to `yadc/api/events.py` with `envs: list[str]` field
- Added `@event_handler(EnvironmentsChangedEvent)` to `SSEEvents` in `sse_events.py`
- Created `EnvWatcherService` (`yadc/api/modules/env_watcher.py`) using `watchdog` to monitor `CONFIG_PATH/config.toml`
  - 1-second debounce to coalesce rapid bursts
  - Dispatches `EnvironmentsChangedEvent` with all env names on create/modify/delete/move of `config.toml`

**Frontend:**
- Added `EnvironmentsChangedEventZ` schema to `stores/events.ts`
- Subscribed to `environments_changed` events in the SSE `connect()` listener and called `refreshEnvs()` automatically
- Removed the manual "reload when settings dialog closes" effect from `EnvSelector.svelte` — now redundant
- Also calls `refreshEnvs()` on `resumption_failed` as a safety net for missed events during disconnection

### Templates

**Backend:**
- Added `TemplatesChangedEvent` to `yadc/api/events.py` with `templates: list[str]` field
- Added `@event_handler(TemplatesChangedEvent)` to `SSEEvents` in `sse_events.py`
- Created `TemplateWatcherService` (`yadc/api/modules/template_watcher.py`) using `watchdog` to monitor `STATE_PATH/templates` for `*.jinja` files
  - 1-second debounce
  - Dispatches `TemplatesChangedEvent` with all current template names (via `cmd_templates.list_user_template()`)

**Frontend:**
- Added `TemplatesChangedEventZ` schema to `stores/events.ts`
- Subscribed to `templates_changed` events and called `refreshTemplates()` automatically

### Notes
- Multi-tab sync: editing envs/templates in one tab updates all other tabs automatically
- CLI changes are detected: running `yadc envs set ...` or editing `.jinja` files directly pushes live updates
- Frontend `fetchEnvs` and `fetchTemplates` are debounced (25ms), so overlapping manual refresh + SSE refresh collapse to one request

**Files:** `yadc/api/events.py`, `yadc/api/modules/sse_events.py`, `yadc/api/modules/env_watcher.py`, `yadc/api/modules/template_watcher.py`, `yadc/api/modules/__init__.py`, `yadc/webui/src/lib/stores/events.ts`, `yadc/webui/src/lib/components/settings/EnvSelector.svelte`

## Phase 2: Dataset Page Request Coordination ✅ Implemented

**Validated finding:** The original "4 requests per dataset page" claim was overstated. Log analysis shows the actual per-dataset requests are:

1. `GET /api/datasets/{name}/images?limit=50` — image grid
2. `GET /api/datasets/{name}/caption` — captioning status (mid-caption poll)
3. `GET /api/configs/{name}` — config tab **(fires twice from two sibling components)**

`GET /api/datasets` is the global list (used by the topbar), not per-dataset. The real problem is **uncoordinated fetches + a duplicated config request** from `DatasetConfig.svelte` and `CaptionSettings.svelte` both independently calling `fetchConfig()`.

**Approach:** Store-level async debounce with dedupe (no caching, no dedicated store object).

**Backend:** No changes needed.

**Frontend:**
- Replaced `debounce` in `async.ts` with an async version that:
  - Delays execution by `delay` ms (standard debounce)
  - Returns a shared promise for calls with the same args while a previous invocation is in-flight (dedupe)
  - Cancels the pending timer when called with different args
- Applied debounce to core API functions:
  - `fetchEnvs` in `envs.ts` — dedupes simultaneous manual refreshes and future SSE-driven refreshes
  - `fetchConfig` in `configs.ts` — both `DatasetConfig` and `CaptionSettings` now collapse to one request
  - `fetchImagesDebounced` in `datasetImages.ts` — used for initial page loads; `loadMore` still uses the original
  - `fetchCaptioningStatusDebounced` in `datasetImages.ts` — used for the initial status poll
  - `fetchDatasets` in `datasetImages.ts` — dedupes rapid navigation between home and dataset pages
  - `fetchCaption` in `datasetImages.ts` — dedupes rapid image selection changes (arrow keys / rapid clicks)
  - `fetchHistory` in `datasetImages.ts` — dedupes rapid image selection changes
  - `fetchModels` in `envs.ts` — rapid env selection changes in `EnvSelector` collapse to one call
  - `fetchTemplates` and `fetchTemplate` in `templates.ts` — simultaneous component loads and rapid selection changes deduped
- Updated `+page.svelte` to use `fetchImagesDebounced` and `fetchCaptioningStatusDebounced` in the dataset change effect
- Default debounce delay is **25 ms** (configurable via `PUBLIC_API_DEFAULT_DEBOUNCE_MS` in `yadc/webui/.env`)

**Intentionally skipped:**
- A unified `datasetPageStore` — the debounce solves the duplicate + rapid-nav problems with less code
- Prop-drilling config — debounce dedupe achieves the same result transparently

**Files:** `yadc/webui/src/lib/async.ts`, `yadc/webui/src/lib/stores/configs.ts`, `yadc/webui/src/lib/stores/datasetImages.ts`, `yadc/webui/src/routes/datasets/[name]/+page.svelte`

## Phase 3: SSE Caption Text ✅ Implemented

**Backend:**
- Add `caption: str = ""` field to `ImageCaptionedEvent`
- In `CaptioningService._emit_image_captioned()`, pass `dataset_image.caption` to the event

**Frontend:**
- Add `caption: string` to `ImageCaptionedEventZ` schema and `ImageCaptionedEvent` type
- Update `stores/events.ts` to store received captions in a `Map<number, string>` keyed by image ID
- Add `getStoredCaption(imageId)` helper to the events store
- Update `ImageDetail.svelte` to check the stored caption before calling `fetchCaption()` — if the caption was received via SSE, skip the HTTP call entirely

**Files:** `yadc/api/events.py`, `yadc/api/services/captioning.py`, `yadc/webui/src/lib/stores/events.ts`, `yadc/webui/src/lib/components/dataset/ImageDetail.svelte`

## Phase 4: Config Caching (Nice-to-Have)

Intentionally skipped. The debounce on `fetchConfig` already solved the duplicate fetch problem. Adding a cache layer introduces invalidation complexity without meaningful benefit for this use case.

## Final Notes

All phases completed. The project now uses a consistent debounce pattern across all stores:
- Private `_fetchXxx()` for the raw implementation
- Public `fetchXxx = debounce(_fetchXxx)` for the debounced export

The default debounce delay (25ms) is configurable via `PUBLIC_API_DEFAULT_DEBOUNCE_MS` in `yadc/webui/.env`.

All SSE thin events use the filesystem watcher pattern (`EnvWatcherService`, `TemplateWatcherService`) for consistent CLI+webui change detection.

## Request Count Comparison

| Workflow | Before | After (Phase 1+2+3) |
|----------|--------|---------------------|
| Open dataset page | 3 requests (images + caption status + configs×2) | 2 requests (images + caption status; config shared) |
| Rapid tab-switching | N×3 requests | Debounced to 1 load per dataset |
| Rapid image selection | N caption+history requests | Debounced to 1 per image |
| Switch env in dropdown | N model list requests | Debounced to 1 per env |
| Open CaptionSettings panel | 1 request (env list + 1 for details) | 0 requests (env details already loaded) |
| Select captioned image | 1 request (fetchCaption) | 0 requests (caption from SSE) |
| Edit env (webui or CLI) | Manual refresh required | Auto-sync via SSE to all tabs |
| Edit template (webui or CLI) | Manual refresh required | Auto-sync via SSE to all tabs |
| **Total per dataset session** | **~5+ requests** | **~2 requests** |

## Cleanup (2026-05-28)

Addressed review findings from the 22 temporary commits (22ef40f..59b49f1).

### Done

1. **`debounce()` rewritten** — now properly resets the timer on each call (true debounce) while keeping dedupe for in-flight promises. Entries are deleted before resolve/reject to prevent stale reuse. Added SSR guard (`if (!browser)` pass-through).

2. **Watcher base class extracted** — new `yadc/api/watcher_base.py` with `SinglePathWatcherService` (shared debounce, observer lifecycle, directory-existence check). `env_watcher.py` and `template_watcher.py` reduced from ~115 lines each to ~38 lines (just constructor args).

3. **`resumption_failed` handler** — now calls both `refreshEnvs()` and `refreshTemplates()`.

4. **True LRU in `_storedCaptions`** — `getStoredCaption()` promotes accessed entries (delete + re-insert). `MAX_STORED_CAPTIONS` reduced to 64 with a doc comment noting the feature may be refined.

5. **`.env.example` added** — documents `PUBLIC_API_DEFAULT_DEBOUNCE_MS=25`.

6. **`fetchEnv()` export** — removed since `fetchEnvs` is the only one being used now.

7. **Enriched SSE event payloads** — `EnvironmentsChangedEvent` and `TemplatesChangedEvent` now include `envs: list[str]` and `templates: list[str]` respectively. The watcher base class was extended to accumulate changed file paths during debounce and delegate event creation to subclasses via `create_event(changed_files)`. Frontend Zod schemas updated to match.

### Open Questions

- For Phase 3, should the caption be included in `ImageCaptionStartedEvent` as well (for the spinner state)? Probably not needed since the user hasn't selected the image yet.
