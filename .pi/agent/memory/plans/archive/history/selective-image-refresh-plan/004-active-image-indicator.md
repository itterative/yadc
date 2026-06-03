---
date: 2026-05-27
---
# Active Image Captioning Indicator

**Context:** During captioning, users could only see which images *finished* (via tile-flash animation). There was no indication of which image was actively being processed by the API — a gap of several seconds per image where nothing appeared to be happening.

**Decision:** Added `ImageCaptionStartedEvent` emitted from the backend just before the API call. Frontend shows a diagonal gradient shimmer sweep on the active tile using a `::after` pseudo-element.

**Rationale:** The "derive from progress count" approach was ruled out because the frontend's image list may be paginated, filtered, or differently ordered from the backend's processing order. A dedicated event is precise and reliable.

**Implementation details:**
- Backend: `ImageCaptionStartedEvent(dataset_name, job_id, image_id, file_name)` dispatched before `_acaption_one()`. One extra `get_image_by_path` DB query per image (cheap, indexed).
- Frontend: `currentlyCaptioning` store auto-clears on `image_captioned`, `image_caption_error`, or terminal `captioning_status` events.
- Animation: `::after` pseudo-element with `linear-gradient(110deg, ...)` sweeping from 150% to -150% over 3.5s. Accent color at 20-35% opacity. Uses `color-mix(in oklch, ...)` for the gradient bands.

**Files touched:** `events.py`, `captioning.py`, `sse_events.py`, `events.ts`, `DatasetImage.svelte`, `DatasetBrowser.svelte`, `+page.svelte`
