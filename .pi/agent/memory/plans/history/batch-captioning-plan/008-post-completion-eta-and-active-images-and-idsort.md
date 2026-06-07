---
name: 008-post-completion-eta-and-active-images-and-idsort
description: Post-completion follow-ups after the 4-phase batch-captioning plan landed — ETA correctness under concurrency, multi-active-images UI, and deterministic id-DESC processing order.
category: meta
---

# Post-completion follow-ups

After the 4-phase plan (Phases 1–4) was committed, three related issues
surfaced that all stem from the same root cause: the existing UI/state
shapes were written for sequential (N=1) captioning and break under
`max_concurrent > 1`. Each was a small, focused fix.

## 1. ETA fix (Backend + Frontend)

**Symptom**: ETA on the topbar used `remaining × avg per image ms / 1000`,
which is the per-image *wall-clock* time × N images — but with
`max_concurrent > 1`, actual wall-clock is that figure divided by
`max_concurrent`. The estimate was wildly pessimistic for parallel runs.

**Fix**:
- **Backend** (`yadc/api/events.py`): `CaptioningStatusEvent` gets
  `elapsed: float = 0.0` and `max_concurrent: int = 1`.
- **Backend** (`yadc/api/services/captioning.py`):
  `AsyncCaptionJob._started_at = time.monotonic()` in `_arun`;
  `_emit_status` and `_snapshot_locked` compute `elapsed = monotonic -
  _started_at` and include both fields. `JobInfo` gets the same two
  fields.
- **Frontend** (`yadc/webui/src/lib/eta.ts`, new): pure helper
  `computeEtaSeconds({ total, processed, ring, maxConcurrent })` that
  smooths the per-model `captionTimingRing` with an EMA (α=0.3, fold
  over the ring buffer; no new persisted store) and divides by
  `maxConcurrent` for the throughput.
- **Frontend** (`yadc/webui/src/routes/datasets/[name]/DatasetTopbar.svelte`):
  uses the new helper; displays `elapsed` and `max_concurrent` (only
  when >1) alongside the ETA.

**Tests** (`tests/api/test_captioning_unit.py`):
- Status event includes `max_concurrent` from options.
- Status event includes non-zero `elapsed` once started.
- Status event has `elapsed=0` before `_arun` runs.

## 2. Multi-active-images UI (Frontend only)

**Symptom**: With `max_concurrent > 1`, multiple images are in flight
but `currentlyCaptioning` was a single `writable<{dataset_name,
image_id} | null>`. Each new `image_caption_started` event *overwrote*
the previous, so the shimmer animation showed on at most one image
regardless of concurrency. Per-image events arriving out of submission
order made the bug visible to the user.

**Fix**: `_currentlyCaptioning` becomes a
`writable<ReadonlyMap<string, CaptioningTarget>>` keyed by
``${dataset_name}#${image_id}`` (string keys so `Set`/`Map` `has` and
`delete` work; a plain `Set<{...}>` fails because fresh object
literals have different identity). The public `currentlyCaptioning`
store is a `derived` that exposes a `ReadonlySet<CaptioningTarget>`.

- `events.ts`: add `addCurrentlyCaptioning()` and
  `clearCurrentlyCaptioning()` helpers; update the 3 SSE handlers
  (started/captioned/error) and the `captioning_status` terminal
  cleanup to use the map.
- `+page.svelte`: `captioningImageIds: SvelteSet<number>` (filtered to
  current dataset) passed to `DatasetBrowser`.
- `DatasetBrowser.svelte`: `captioningIds: ReadonlySet<number>` prop
  (was `captioningId: number | null`); passes
  `captioningIds.has(item.id)` to each `DatasetImage.captioning`.
- `ImageDetail.svelte`: `isCaptioning` derived iterates the set
  (O(max_concurrent), trivial).
- `actions.ts`: 2 `setCurrentlyCaptioning({...})` calls replaced with
  `addCurrentlyCaptioning(ds, id)`.

**Edge cases handled**:
- Cancellation (in-flight tasks don't fire per-image events) →
  terminal-status handler sweeps the set for the dataset.
- Lost/replayed events on reconnect → add/remove are idempotent
  no-ops on missing entries.
- Cross-dataset isolation → set is global; the page filters to the
  current dataset before passing to the browser.

## 3. id-DESC sort for predictable parallel order (Backend)

**Symptom**: The runner uses `CaptioningRunner.caption_images()`,
which calls `resolve_dataset` (filesystem iteration order — not
deterministic across platforms) and then submits tasks in that order.
With `max_concurrent > 1`, completion order depends on per-image
latency variance, making the visible "captioning" pattern feel
random.

**User intent (confirmed)**: caption the **newest** images first —
database `id` descending. The WebUI's masonry grid handles its own
visual layout (column packing), so we don't need to match the SQL
order; we just need a deterministic id-DESC submission order so the
user gets a predictable "newest first" experience.

**Fix** (two iterations — see "Iteration 2" below for the final shape):
- **Loader** (`yadc/core/captioning/loader.py`): no change to its
  public interface. Stays generic — returns the
  filesystem-iteration order, leaving the caller to reorder if it
  cares. (An earlier draft added an `image_id_resolver` parameter
  for a Python sort; this was removed in the SQL refactor.)
- **Repository** (`yadc/api/services/dataset_repository.py`): new
  `list_image_paths_desc(dataset_name) -> list[tuple[str, int]]`
  method. Single SQL query: `ORDER BY di.id DESC`. No N+1.
- **Service** (`yadc/api/services/datasets.py`): new
  `get_image_paths_in_desc_order(dataset_name) -> list[str]`
  pass-through.
- **API service** (`yadc/api/services/captioning.py`):
  `preflight_images` calls `get_image_paths_in_desc_order` once
  and reorders the loader's filesystem-resolved result by building
  a `path → position` map and sorting with a sentinel for
  not-in-DB paths. Removes the earlier `_image_id_resolver`
  method.

### Iteration 2: SQL ORDER BY everywhere

After the first iteration landed with a Python sort in the loader
(via an `image_id_resolver` callback), the user requested
"`ORDER BY rowid DESC` for both" — the listing endpoint and the
captioning. The Python sort was moved to a dedicated SQL query so
both paths use real `ORDER BY di.id DESC` in SQL.

**Listing endpoint** (`yadc/api/services/dataset_repository.py:174`
+ service + controller): `list_images` SQL changed from
`WHERE id > ? ORDER BY id ASC` to `WHERE id < ? ORDER BY id DESC`.
The cursor param renamed `after_id` → `before_id` (matching the
existing `before_id` precedent in `config_history`).
`DatasetService.list_images` now takes `before_id: int | None` and
translates `None` to `2**63 - 1` (effectively unbounded) so the
first page returns the newest images. The `next_token` returned
to the client is the smallest id on the current page (the last
image in DESC order) — pass it as the next `before_id` to get
older images.

**Frontend** (`yadc/webui/src/lib/stores/dataset/api.ts` +
`+page.svelte`): `_fetchImages` option renamed `afterId` →
`beforeId`; query param `after_id` → `before_id`. No semantic
change to the page (it just keeps appending pages to the existing
`images` array and passing the new `nextToken` as the next
`beforeId`).

**Captioning**: the earlier N+1 `_image_id_resolver` (one
`get_image_by_path` per image) was replaced with a single
`get_image_paths_in_desc_order` SQL call. Same end-state (id
DESC), but no Python sort, no N+1, and the SQL `ORDER BY` is
literal rather than implicit through the resolver key.

**Performance**: One SQL query per preflight (returns all
`(path, id)` for the dataset) replaces N queries. For 10k images,
both are fast on SQLite (~10ms either way) but the single-query
version is also cleaner.

**Tests**:
- `tests/api/test_dataset_repository.py`:
  - `TestListImages` — 3 tests updated to DESC order: first page
    (large `before_id`) returns the newest images, `before_id`
    cursor returns id < cursor in DESC, empty dataset.
  - `TestListImagePathsDesc` (new, 2 tests) — single-query path
    list in DESC, empty dataset.
- `tests/api/test_captioning_unit.py`:
  - `test_caption_images_receives_desc_sorted_to_do` —
    `preflight_images` reorders via the new SQL helper, passes
    the sorted list to `caption_images`, calls
    `get_image_paths_in_desc_order` exactly once.
- `tests/core/captioning/test_loader.py`:
  - 3 tests for the now-removed `image_id_resolver` parameter
    were deleted. The loader is back to having no
    order-related parameters — the API service owns the reorder.

## See also

- `architecture-overview` — code structure
- `docs/captioning-workflow` — shared runner + per-side concerns
- `docs/captioning-runner` — runner interface (unchanged here)
- The original 4-phase plan entries (`001`–`007`) in this directory
