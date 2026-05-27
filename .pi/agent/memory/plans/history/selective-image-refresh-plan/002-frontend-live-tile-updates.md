---
date: 2026-05-27
---

# Phase 2: Frontend SSE parsing + live tile updates

**Context:** Backend now emits `image_captioned` and `image_caption_error` SSE events per image. The frontend needed to consume these and update grid tiles in-place during captioning.

**Decision:** Added Zod schemas and writable stores for both event types in `events.ts`. The dataset page subscribes via `$effect` and does in-place `images.map()` updates on matching events. `ImageInfo` gained an optional `caption_error` field. `DatasetImage.svelte` shows a ⚠ warning badge when set. The old NOTE about grid tiles not updating live was replaced with a comment describing the new behavior.

**Rationale:** Using stores + `$effect` is the established pattern in this codebase (matches how `captioningStatus` etc. are consumed). The `images.map()` approach is simple and works well since the grid renders from the `images` array reactively. The full reload on `handleCaptioningDone` is kept as a safety net for missed events.

**Files touched:** `yadc/webui/src/lib/stores/events.ts`, `yadc/webui/src/lib/stores/datasetImages.ts`, `yadc/webui/src/lib/components/dataset/DatasetImage.svelte`, `yadc/webui/src/routes/datasets/[name]/+page.svelte`
