---
date: 2026-05-27
---

# Phase 1: Backend per-image events

**Context:** During captioning the grid showed stale data until the entire batch finished. SSE `CaptioningStatusEvent` only carried aggregate counts. A note in the dataset page (+page.svelte line 59) documented this limitation.

**Decision:** Emit per-image SSE events after each image is captioned or fails. Two new event types: `ImageCaptionedEvent` (flattened `ImageInfo` fields + `dataset_name`/`job_id`) and `ImageCaptionErrorEvent` (`image_id` + `error`). Flattened fields instead of nesting an `ImageInfo` object to avoid an import cycle between `events.py` and `datasets.py`.

**Rationale:** The event shape mirrors the `GET /datasets/<name>/images` response for a single image, so the frontend can update tiles using the same `ImageInfo` type without additional fetching.

**Files touched:** `yadc/api/events.py`, `yadc/api/services/datasets.py`, `yadc/api/services/captioning.py`, `yadc/api/modules/sse_events.py`
