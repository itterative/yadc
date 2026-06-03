---
date: 2026-05-28
---
# Phase 2 Refined: Dataset Page Request Coordination

**Context:** The original Phase 2 proposed two options: a dedicated `/browser` backend endpoint (Option A) or a frontend store with TTL caching (Option B). The "4 requests per dataset page" claim was the main driver.

**Validated finding:** Log analysis showed the actual per-dataset requests are smaller and different:
- `GET /api/datasets/{name}/images?limit=50` — image grid
- `GET /api/datasets/{name}/caption` — captioning status poll
- `GET /api/configs/{name}` — **fires twice** from `DatasetConfig.svelte` and `CaptionSettings.svelte`

`GET /api/datasets` is the global list (used by topbar), not per-dataset. The real problem is uncoordinated fetches + a duplicated config request.

**Decision:** Replace both Option A and Option B with a simpler approach: a `datasetPageStore` that coordinates `fetchImages`, `fetchCaptioningStatus`, and `fetchConfig` into a single reactive state. Config is fetched once and passed down to both consumers as a prop. A 50–100ms debounce prevents redundant loads during rapid tab-switching. Caching is intentionally excluded per user preference.

**Rationale:** The original Options A/B were designed around a "4 requests" problem that didn't fully exist. After log validation, the scope is smaller and more targeted: coordinate what exists rather than build new endpoints or cache layers. No backend changes needed.

**Files to touch:** `yadc/webui/src/lib/stores/datasetImages.ts`, `yadc/webui/src/routes/datasets/[name]/+page.svelte`, `yadc/webui/src/routes/datasets/[name]/SidePanel.svelte`, `yadc/webui/src/routes/datasets/[name]/DatasetConfig.svelte`, `yadc/webui/src/routes/datasets/[name]/CaptionSettings.svelte`
