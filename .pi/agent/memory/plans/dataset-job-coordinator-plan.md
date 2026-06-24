---
name: dataset-job-coordinator-plan
description: Cross-service per-dataset job mutex (DatasetJobService) so captioning and tagging can never overlap on the same dataset. Extensible to future sidecar-writing job kinds.
last_history: 0
---

# Dataset Job Coordinator

## Status
Complete.

## Why
Captioning and tagging both mutate image sidecars. Letting them overlap on the
same dataset races on writes (a caption run could clobber a tag extras-merge, or
vice versa). Each service already rejected *same-type* concurrency but neither
knew about the other. We need one authoritative gate across all sidecar-writing
job kinds, extensible to future features (export-with-write, etc.) without the
services coupling to each other.

## Design
`DatasetJobService` (`yadc/api/services/dataset_jobs.py`) — a `Service`
auto-discovered singleton, the single source of truth for "who holds a dataset".
API:

- `try_acquire(dataset_name, kind, job_id) -> JobClaim` — atomic claim; raises
  `DatasetBusyError(ValueError)` if held.
- `release(claim)` — identity-checked (a stale handle can't evict a newer owner).
- `current(dataset_name)` / `is_busy(dataset_name)` — non-locking advisory reads.

`DatasetBusyError` subclasses `ValueError` so controllers that already map
`ValueError → 409` keep working; the single-tag controller catches the subclass
for an explicit 409.

### Integration points
1. **CaptioningService** — injects `DatasetJobService`; acquires the claim in
   `start_job_async` after preflight succeeds & there's work (before `job.start()`),
   releases at the top of `_cleanup_async` (before the status-visibility grace
   sleep, so the dataset frees as soon as the task ends).
2. **TaggingService.start_tag_job_async** — acquires before creating the task;
   `_TagJobState.claim` carries the handle; released in `_run_tag_job` finally.
3. **Single-image tag** — `tag_image` stays claim-free (the batch job calls it
   internally, so it can't own the claim). New `tag_single_image_async` wrapper
   does `try_acquire → tag_image → release` in finally; the controller's
   `POST .../images/<id>/tag` switches to it and gains `except DatasetBusyError → 409`.
4. **Single-image caption / refine** — already go through `start_job_async`, so
   they're covered automatically.

### Concurrency
Each service's own lock serializes same-type starts (fast path + better error
messages). The coordinator's single `asyncio.Lock` makes `try_acquire` the
decisive atomic point across all kinds — no window where two cross-type starts
both succeed.

## Scope chosen
Batch + single-image jobs both gated (both write sidecars). Interactive manual
edits (frontend caption/tag edits via `DatasetService` directly) are NOT
"gated jobs" — they go through `DatasetService` and are out of scope.

## Tests
`tests/api/test_dataset_jobs.py` (14): acquire/release, cross-kind + same-kind
conflict, `DatasetBusyError` is a `ValueError`, stale/wrong-claim release is a
no-op, two concurrent racers yield exactly one winner, single-tag releases on
success + 409s when a batch holds. Existing service fixtures updated to inject
a real `DatasetJobService`.

## Future (out of scope)
- Frontend "dataset busy" indicator / cross-service status surface.
- Manual claim-recovery admin endpoint (process crash loses all in-memory state
  anyway; fresh start = empty registry).
- Queueing single-image ops instead of hard 409.
