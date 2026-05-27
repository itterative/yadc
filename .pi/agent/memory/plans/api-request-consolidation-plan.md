---
name: api-request-consolidation-plan
description: Consolidate redundant API requests in the webui — environment list/details, dataset browser (4→1), and SSE caption text
---

# API Request Consolidation

## Problem Statement

The webui makes more API requests than necessary in several common workflows:

1. **Environment list vs details**: `GET /api/envs` returns only names, so `EnvSelector` must call `GET /api/envs/{name}` individually after the user selects one.
2. **Dataset browser loads 4 requests**: Loading a dataset page triggers `GET /images`, `GET /datasets`, `GET /caption`, and `GET /configs/{name}` in parallel.
3. **Environment call repeats**: Opening CaptionSettings on a dataset page re-triggers the environment fetch (same as #1).
4. **SSE caption text missing**: `ImageCaptionedEvent` sends image metadata but not the caption text, so selecting a captioned image still requires `GET /images/{id}/caption`.

## Phase 1: Environment Details Endpoint

**Backend:**
- Add `GET /api/envs/details` → returns `{[name: string]: EnvInfo}` (all envs with full details in one call)
- Keep `GET /api/envs` returning `list[str]` for backward compatibility (settings dialog)

**Frontend:**
- Add `fetchEnvsDetails()` to `stores/envs.ts`
- Update `EnvSelector.svelte` to call `fetchEnvsDetails()` once on mount, populate the dropdown from keys, and pre-fill detail fields for the default selection
- Remove the per-env `fetchEnv()` call from the selection effect (keep as fallback)

**Files:** `yadc/api/controllers/api_envs.py`, `yadc/webui/src/lib/stores/envs.ts`, `yadc/webui/src/lib/components/settings/EnvSelector.svelte`

## Phase 2: Dataset Browser Consolidation

Two approaches — pick one (or implement both as Phase 2A/2B).

### Option A: Dedicated Browser Endpoint (single-request)

**Backend:**
- Add `GET /api/datasets/{name}/browser` → returns `{ images: ImagePage, dataset: DatasetInfo, captioning_status: CaptioningJobInfo }`
- Combines the 3 requests that fire on every dataset page load into one

**Frontend:**
- Add `fetchDatasetBrowser()` to `stores/datasetImages.ts`
- Update `datasets/[name]/+page.svelte` to call `fetchDatasetBrowser()` in `onMount` instead of the three separate calls
- Keep `GET /configs/{name}` as a separate call (changes less frequently; could be cached in Phase 3)

### Option B: Frontend Store + Debounce (no backend changes)

**Frontend:**
- Create a `datasetBrowserStore` (in `stores/datasetImages.ts`) that manages the combined state:
  ```ts
  interface DatasetBrowserState {
    images: ImageInfo[];
    dataset: DatasetInfo | null;
    captioningStatus: CaptioningJobInfo | null;
    isLoading: boolean;
    error: string | null;
  }
  ```
- On dataset name change, fire all three fetches in parallel (`fetchImages` + `fetchDatasets` + `fetchCaptioningStatus`) and combine results
- Add a short debounce (e.g. 100ms) to the dataset name effect so rapid navigation doesn't trigger redundant loads
- Cache results in the store keyed by dataset name — navigating away and back reuses the cached data until it expires (TTL ~30s)
- The store's `load()` method returns a promise so callers can `await` completion
- Keep `GET /configs/{name}` as a separate call (or cache it in Phase 3)

**Pros vs Option A:**
- No backend changes needed
- Store can be reused by other pages/components that need combined dataset state
- Debounce + TTL caching handles the "duplicate request" problem without a new endpoint

**Cons vs Option A:**
- Still 3 HTTP requests (just parallel instead of scattered)
- No single round-trip optimization
- Store lifecycle management (cleanup, invalidation) adds complexity

**Files (Option A):** `yadc/api/controllers/api_datasets.py`, `yadc/webui/src/lib/stores/datasetImages.ts`, `yadc/webui/src/routes/datasets/[name]/+page.svelte`
**Files (Option B):** `yadc/webui/src/lib/stores/datasetImages.ts`, `yadc/webui/src/routes/datasets/[name]/+page.svelte`

## Phase 3: SSE Caption Text

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

- Cache `GET /configs/{name}` in a module-level store with a short TTL (e.g. 5 minutes)
- CaptionSettings checks the cache first, only fetching if stale
- Invalidate cache on `PATCH /configs/{name}` success

## Request Count Comparison

| Workflow | Before | After (Phase 1+2+3) |
|----------|--------|---------------------|
| Open dataset page | 4 requests | 1 request (browser endpoint) |
| Open CaptionSettings panel | 1 request (env list + 1 for details) | 0 requests (env details already loaded) |
| Select captioned image | 1 request (fetchCaption) | 0 requests (caption from SSE) |
| **Total per dataset session** | **~6+ requests** | **~1 request (Option A)** / **~2 requests (Option B)** |

## Open Questions

- **Phase 2: Option A vs Option B?** Option A (single endpoint) is cleaner but requires backend changes. Option B (store + debounce) avoids backend work and handles duplicate requests via caching. Could implement both and let the user choose.
- Should `GET /api/envs/details` also be available as a query param on the existing `GET /api/envs` (e.g. `?details=true`)? This would be simpler but changes the response shape.
- For Phase 3, should the caption be included in `ImageCaptionStartedEvent` as well (for the spinner state)? Probably not needed since the user hasn't selected the image yet.
- Should the browser endpoint (Option A) include the config defaults too, reducing to a single request for Phase 1+2? This would make the endpoint quite large — better to keep config separate for now.
