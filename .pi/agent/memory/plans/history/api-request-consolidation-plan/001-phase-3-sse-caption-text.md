---
date: 2026-05-28
---
# Phase 3: SSE Caption Text

**Context:** When a single image is captioned (or a batch finishes), the `ImageCaptionedEvent` only carried image metadata. The `ImageDetail.svelte` component still had to call `GET /images/{id}/caption` to display the new caption, even though the server had just generated it.

**Decision:** Add a `caption` field to `ImageCaptionedEvent` and cache received captions on the frontend. The `ImageDetail.svelte` component now checks the SSE cache first after a caption job completes, skipping the redundant HTTP round-trip entirely.

**Rationale:** This was the smallest change with the most immediate user impact — no spinner when opening a freshly-captioned image. A bounded `Map<number, string>` (LRU, max 1000) avoids unbounded memory growth. Stored captions are cleared after an explicit `fetchCaption()` to prevent stale data.

**Files touched:**
- `yadc/api/events.py`
- `yadc/api/services/captioning.py`
- `yadc/webui/src/lib/stores/events.ts`
- `yadc/webui/src/lib/components/dataset/ImageDetail.svelte`
