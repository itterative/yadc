---
name: selective-image-refresh-plan
description: Per-image SSE events during captioning for live grid tile updates.
last_history: 3
---

# Selective Image Refresh During Captioning

## Problem

During captioning, the grid shows stale data until the entire batch finishes. Users can't see which images have been processed until the final full reload (`handleCaptioningDone`). The SSE `CaptioningStatusEvent` only carries aggregate counts (processed/total/errors), not individual image IDs.

## Solution

Emit per-image SSE events after each image is captioned (or fails). The event carries the full `ImageInfo` data (same shape as the `GET /datasets/<name>/images` API response) so the frontend can update individual grid tiles without a full reload. The same mechanism works for both batch and single-image captioning.

## Design

### Event types

Two new event types in `yadc/api/events.py`:

1. **`ImageCaptionedEvent`** — emitted after an image is successfully captioned
   - `dataset_name: str`
   - `job_id: str`
   - Flattened `ImageInfo` fields: `id`, `file_name`, `path`, `has_caption`, `has_toml`, `width`, `height`, `draft_names`, `last_modified_t`

2. **`ImageCaptionErrorEvent`** — emitted when captioning a single image fails
   - `dataset_name: str`
   - `job_id: str`
   - `image_id: int`
   - `error: str`

### Backend changes

#### 1. New events (`yadc/api/events.py`)

- Add `ImageCaptionedEvent` and `ImageCaptionErrorEvent` dataclasses
- `ImageCaptionedEvent` embeds the same fields as `ImageInfo` flattened directly on the event (avoids import cycle with `datasets.py`)

#### 2. CaptionJob emits per-image events (`yadc/api/services/captioning.py`)

- After `_caption_one()` succeeds in the captioning loop:
  1. Call `self._dataset_service.refresh_image_index(dataset_name, image_id)` to update SQLite
  2. Call `self._dataset_service.get_image(dataset_name, image_id)` to get fresh `ImageInfo`
  3. Dispatch `ImageCaptionedEvent(dataset_name, job_id, image_info)`

- When `_caption_one()` fails (in the `except` block):
  1. Dispatch `ImageCaptionErrorEvent(dataset_name, job_id, image_id, error_message)`

- **Need image_id lookup**: `CaptionJob` operates on `DatasetImage` (core model, filesystem paths). It needs to map the `DatasetImage.path` back to a SQLite `image_id`. Options:
  - **A)** Add a method `DatasetService.get_image_by_path(dataset_name, path) -> ImageInfo | None` — query by path. Simple and clean.
  - **B)** Pre-build a path→id map before the captioning loop starts (query all images once).
  
  **Decision**: Option A — simpler, and the per-image query is cheap (indexed). The captioning loop already hits the DB per image for `refresh_image_index` anyway.

- **New helper**: `DatasetService.get_image_by_path(dataset_name, path) -> ImageInfo | None` — looks up an image row by its filesystem path.

#### 3. SSE dispatch (`yadc/api/modules/sse_events.py`)

- Add `@event_handler` methods for `ImageCaptionedEvent` and `ImageCaptionErrorEvent`
- Both call `self.push(event)` — existing infrastructure handles serialization and delivery

#### 4. Serialization

- `DataclassJSONEncoder` already handles nested dataclasses, so `ImageCaptionedEvent.image` (an `ImageInfo` dataclass) serializes automatically
- The SSE event name will be `image_captioned` / `image_caption_error` (derived from `TYPE` class var)

### Frontend changes

#### 1. Zod schemas (`yadc/webui/src/lib/stores/events.ts`)

- Add `ImageInfoZ` schema matching the backend `ImageInfo` dataclass shape
- Add `ImageCaptionedEventZ` schema: `{ dataset_name, job_id, image: ImageInfoZ }`
- Add `ImageCaptionErrorEventZ` schema: `{ dataset_name, job_id, image_id, error }`

#### 2. SSE listener (`yadc/webui/src/lib/stores/events.ts`)

- Add new writable store `_imageCaptionedEvents` (or expose a callback-based approach)
- **Better approach**: Don't use a store for per-image events. Instead, expose a `onImageCaptioned` callback registration in the events module. The dataset page registers a handler when mounted and unregisters on destroy.
- **Simplest approach**: Add a writable store `lastImageCaptionedEvent` that the dataset page subscribes to. Since events are processed sequentially, each update triggers the subscriber.

**Decision**: Use a writable store `imageCaptionedEvents` as a simple counter/event pair. Actually, the cleanest is to just add the SSE listener inline and directly call the browser store's `updateImage`. But the events module owns the SSE connection...

**Final decision**: Add the SSE listeners in `events.ts` that pipe into two new writable stores:
- `lastCaptionedImage: Readable<{dataset_name: string, job_id: string, image: ImageInfo} | null>` — the most recent captioned image event
- `lastCaptionError: Readable<{dataset_name: string, job_id: string, image_id: number, error: string} | null>` — most recent error event

The dataset page subscribes to these stores in a `$effect` and updates the grid.

#### 3. Dataset page (`yadc/webui/src/routes/datasets/[name]/+page.svelte`)

- Subscribe to `lastCaptionedImage` store
- When event arrives with matching `dataset_name`:
  - Call `browserStore.updateImage(event.image.id, event.image)` to update the tile in-place
  - The `updateImage` function already exists on `createDatasetBrowserStore` and does a shallow merge
- Subscribe to `lastCaptionError` store
  - When event arrives with matching `dataset_name`, mark the tile with a warning state
  - Add an `error` field to the tile's display (simple warning badge for now)

#### 4. DatasetImage tile (`yadc/webui/src/lib/components/dataset/DatasetImage.svelte`)

- Add a `caption_error` prop (optional string)
- When set, show a warning badge (e.g. ⚠) on the tile
- This is the minimal UI for error indication — detailed error display is deferred

### Flow

```
CaptionJob._do_run() loop:
  for img in to_do:
    try:
      _caption_one(model, img, ...)  → saves caption to disk
      refresh_image_index(name, id)  → updates SQLite row
      get_image(name, id)            → reads fresh ImageInfo
      dispatch(ImageCaptionedEvent)  → SSE push → frontend updates tile
    except:
      dispatch(ImageCaptionErrorEvent) → SSE push → frontend shows warning badge

Frontend SSE handler:
  image_captioned → lastCaptionedImage store → dataset page $effect → updateImage()
  image_caption_error → lastCaptionError store → dataset page $effect → updateImage() with error flag
```

### Edge cases

- **Page not open when events fire**: Events are consumed only when the page is mounted. Missed events are harmless — the full reload on `handleCaptioningDone` catches everything.
- **HMR / reconnect**: If SSE reconnects, missed per-image events from the history buffer are replayed. The `updateImage` calls are idempotent.
- **Single-image captioning**: Uses the same `CaptionJob` loop with `image_ids` filter — events flow identically.
- **Draft mode**: When `opts.draft` is set, `has_caption` won't change but `draft_names` will — `refresh_image_index` picks this up.

## Phased implementation

### Phase 1: Backend — events + emission ✅
- Add `ImageCaptionedEvent` and `ImageCaptionErrorEvent` to `events.py`
- Add `DatasetService.get_image_by_path()` helper
- Modify `CaptionJob._do_run()` to dispatch per-image events
- Register handlers in `SSEEvents`

### Phase 2: Frontend — SSE parsing + tile updates ✅
- Add Zod schemas for new events
- Add writable stores and SSE listeners in `events.ts`
- Subscribe in dataset page, call `updateImage()`
- Add warning badge to `DatasetImage.svelte`

### Phase 3: Cleanup
- Remove the NOTE comment about grid tiles not updating live ✅ (done in Phase 2)
- **Conditional `loadInitial` on captioning done** ✅
  - `handleCaptioningDone` now skips `loadInitial()` unless `resumptionFailed` is true (SSE history buffer cycled, some per-image events may have been missed).
  - When `resumptionFailed` is true, a warning toast explains why the grid is refreshing: *"Some captioning events were missed. Refreshing grid to ensure accuracy."*
  - Aggregate dataset stats (`fetchDatasets()`) are always refreshed for the topbar.
  - Pagination is not a concern: `loadInitial` only reloads page 1 (50 images). Per-image events update whatever pages are already loaded; `loadMore()` fetches fresh data when the user scrolls. Images on unloaded pages are irrelevant until scrolled into view.
  - This eliminates the jarring scroll reset that `loadInitial` caused when the user had scrolled beyond page 1.
- Refine the ⚠ warning badge on `DatasetImage.svelte` — current implementation is a minimal first pass. Improve styling, positioning (avoid overlap with draft badge at `top-1 left-1`), and consider clearing the error when the tile is successfully re-captioned or on full reload
- Add a one-shot accent border glow animation (`tile-flash`) that triggers when a tile is updated via per-image events, so the user can see which tiles changed. Uses a `flash` timestamp on `ImageInfo` and `animationend` to auto-clear
- **TODO**: Investigate how to show an animation on the image *currently being processed* (not just after completion). This would require the backend to emit a `captioning_started` per-image event before calling the API, or deriving the current image from the `CaptioningStatusEvent.processed` count and the known image order
- ~~**Pre-existing issue**: If captioning is started from another tab/device, an already-open dataset page may not show the progress bar until the next aggregate `CaptioningStatusEvent`.~~ **RESOLVED — not reproducible.** Traced the SSE stream with curl: `CaptioningStatusEvent(status="running")` is dispatched to all connected clients immediately when a job starts, including cross-tab jobs. `Last-Event-ID` resumption was also verified to work correctly. The symptom was most likely caused by **Hot Module Replacement (HMR)** during development, which tears down and recreates the page (and its `$effect` subscriptions) while captioning is in progress. The existing `fetchCaptioningStatus` poll on page mount already handles recovery from HMR. No backend or frontend changes needed.
- Update todo memory

## Files changed

| File | Change |
|------|--------|
| `yadc/api/events.py` | Add `ImageCaptionedEvent`, `ImageCaptionErrorEvent` |
| `yadc/api/services/datasets.py` | Add `get_image_by_path()` method |
| `yadc/api/services/captioning.py` | Emit per-image events in `CaptionJob._do_run()` |
| `yadc/api/modules/sse_events.py` | Add `@event_handler` for new events |
| `yadc/webui/src/lib/stores/events.ts` | Add Zod schemas, stores, SSE listeners |
| `yadc/webui/src/lib/stores/datasetImages.ts` | Extend `ImageInfo` type with optional `caption_error` |
| `yadc/webui/src/lib/components/dataset/DatasetImage.svelte` | Warning badge for caption errors |
| `yadc/webui/src/routes/datasets/[name]/+page.svelte` | Subscribe to per-image stores, call `updateImage()` |

## Open questions

- ~~Caption text in event or separate fetch?~~ → Full `ImageInfo` shape (same as list API)
- ~~Error events?~~ → Yes, with simple warning badge (detailed UI deferred)
- ~~Single-image captioning same flow?~~ → Yes
