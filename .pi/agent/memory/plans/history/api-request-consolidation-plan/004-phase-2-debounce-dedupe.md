---
date: 2026-05-28
---
# Phase 2: Async Debounce + Dedupe

**Context:** After validating the actual request patterns via server logs, the "4 requests per dataset page" problem was smaller than expected: 2 core requests (images + captioning status) plus a duplicated `GET /api/configs/{name}` from `DatasetConfig.svelte` and `CaptionSettings.svelte` both calling `fetchConfig()` independently.

**Decision:** Instead of building a unified `datasetPageStore` or adding a backend `/browser` endpoint, replaced the sync `debounce` utility with an async version that supports dedupe. Applied it directly to the core API functions.

**Rationale:**
- A dedicated store would add indirection and lifecycle management for marginal benefit
- The async debounce solves both problems at once:
  - **Duplicate config fetch:** Both components call `fetchConfig('cudlil')` → one request, both get the same promise
  - **Rapid tab-switching:** `fetchImagesDebounced('A')` starts a timer, `fetchImagesDebounced('B')` resets it → only B loads
- No prop drilling or component changes needed for config dedupe

**What changed:**
- `async.ts`: `debounce` rewritten as a Map-based async debounce. Each unique serialized argument key gets its own timer and promise. Same-key calls return the existing promise (dedupe). Different keys run independently. Entries are cleaned up after the callback completes. No rejections, no dangling promises.
- `configs.ts`: `fetchConfig` is now `debounce(_fetchConfig, 50)`
- `datasetImages.ts`: Added `fetchImagesDebounced` and `fetchCaptioningStatusDebounced`
- `+page.svelte`: Initial loads use debounced versions; `loadMore` still uses original `fetchImages`

**Files touched:**
- `yadc/webui/src/lib/async.ts`
- `yadc/webui/src/lib/stores/configs.ts`
- `yadc/webui/src/lib/stores/datasetImages.ts`
- `yadc/webui/src/routes/datasets/[name]/+page.svelte`
