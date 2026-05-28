---
date: 2026-05-28
---

# Phase 2 design decisions

**Context:** Phase 1 added `isBatchCaptioning` and `onstopcaptioning` as props threaded through `+page.svelte` → `SidePanel.svelte` → `CaptionSettings.svelte`. We wanted to eliminate this prop drilling and also clean up single-image captioning state tracking.

**Decisions:**

1. **New store module `captionActions.ts`** — holds shared caption logic instead of extending `datasetImages.ts`. Keeps `datasetImages.ts` focused on data fetching. The new store owns `captionOptions` (reactive store written by CaptionSettings, read by action functions), `startBatchCaptioning()`, `captionSingleImage()`, and `stopCaptioning()`.

2. **`stopCaptioning` with internal toasts + optional error handler** — lives in the store, shows toasts on failure, accepts an optional `onError` callback for callers that need to react internally.

3. **Password retry is shared** — `withPasswordRetry()` wraps both single-image and batch captioning calls. It's not a page-level concern.

4. **Caption options are a shared store** — `CaptionSettings` writes assembled options to a reactive store instead of passing them via `onstart` callback. Both `startBatchCaptioning` and `captionSingleImage` read from it.

5. **`isCaptioning` derived from `currentlyCaptioning` store** — `ImageDetail` imports the store and derives `isCaptioning` from `currentlyCaptioning?.dataset_name === datasetName && currentlyCaptioning?.image_id === item.id`. Eliminates `singleImageJobId`, `singleImageTargetId`, `isFocusedImageCaptioning` from `+page.svelte`.

6. **Seed `currentlyCaptioning` for immediate spinner** — both `startBatchCaptioning` and `captionSingleImage` seed the store so the spinner appears before the first SSE event arrives.

7. **Unified completion toast** — one completion `$effect` in `+page.svelte` fires toasts for any job we started (batch or single-image). No toast logic needed in `ImageDetail`.

8. **Remove `oncaptionupdated`** — tile `has_caption` updates already come from `$lastCaptionedImage` SSE effect. The manual update in the completion handler was redundant.

9. **`ImageDetail` calls `captionSingleImage()` directly** — removes `oncaptionimage` prop chain.

**Files touched:**
- New: `yadc/webui/src/lib/stores/captionActions.ts`
- Modify: `yadc/webui/src/routes/datasets/[name]/CaptionSettings.svelte` — write options to store, call actions directly
- Modify: `yadc/webui/src/lib/components/dataset/ImageDetail.svelte` — derive `isCaptioning` from store, call `captionSingleImage` directly
- Modify: `yadc/webui/src/routes/datasets/[name]/SidePanel.svelte` — remove captioning prop threading
- Modify: `yadc/webui/src/routes/datasets/[name]/+page.svelte` — remove single-image state, simplify to batch tracking only
- Modify: `yadc/webui/src/lib/stores/datasetImages.ts` — `stopCaptioning` moves to `captionActions.ts`
