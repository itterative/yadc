---
date: 2026-05-28
---
# Final Debounce Cleanup: Dataset Reads

**Context:** After debouncing envs, templates, configs, and images, three remaining read functions in `datasetImages.ts` were identified as having duplicate-call potential from reactive effects.

**Changes:**
1. `fetchDatasets` (`GET /api/datasets`) — debounced. Called from home page, dataset page effects, and export dialog. Rapid navigation between pages could trigger overlapping fetches.
2. `fetchCaption` (`GET /api/datasets/{name}/images/{id}/caption`) — debounced. Called from `ImageDetail.svelte` `$effect` on every image selection change. Rapid arrow-key navigation or rapid clicking through images now collapses to one fetch for the last-selected image.
3. `fetchHistory` (`GET /api/datasets/{name}/images/{id}/history`) — debounced. Same pattern as `fetchCaption`, called on image selection change and history expand.

**Intentionally not debounced:**
- `fetchExportBackends`, `fetchDatasetDrafts`, `fetchPromptPreview` — single-shot explicit user actions
- `fetchImages` (original) — kept for `loadMore` infinite scroll
- `fetchCaptioningStatus` (original) — kept for polling
- All mutating functions (save, update, delete, start/stop captioning, etc.)

**Files touched:** `yadc/webui/src/lib/stores/datasetImages.ts`
