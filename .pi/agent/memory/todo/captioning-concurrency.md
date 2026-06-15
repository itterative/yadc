---
name: captioning-concurrency
description: Plan for limiting concurrent captioning jobs across datasets. Queue-based vs shared-captioner analysis.
category: backend
priority: 2
---

# Captioning Cross-Dataset Concurrency

## Current state

- `CaptioningService._async_jobs` is `dict[str, AsyncCaptionJob]` — one job per dataset, no global limit
- Each job creates its own `CaptioningRunner` → `AsyncSession` + `APICaptioner` (async context manager lifecycle)
- Per-job `max_concurrent` controls image-level parallelism within a single dataset via `asyncio.Semaphore` inside `CaptioningRunner.caption_images()`
- Starting N dataset jobs means N concurrent API connections, no backpressure

## Recommended: Queue-based approach (1–2 hours)

Add a global concurrency gate to `CaptioningService` so jobs wait for a free slot:

1. **Global `asyncio.Semaphore`** in `CaptioningService.__init__` — `configuration.max_concurrent_jobs`
2. **New `JobStatus`: `"queued"`** — frontend shows a waiting indicator
3. **Job acquires semaphore in `_ado_run`** before building the runner; releases on exit
4. Each job still gets its own connection (httpx connections are lightweight), just capped globally

Pros: minimal changes, doesn't touch core captioning layer, composes with per-job `max_concurrent`.

## Alternative: Shared captioner across datasets (3–5 sessions)

Reuse one `AsyncSession` + `APICaptioner` across multiple datasets' jobs.

Would require:

- **Decouple session lifecycle from job lifecycle** — session pool or long-lived shared runner
- **Change `CaptioningRunner`** — accept pre-existing sessions instead of creating/destroying per job
- **Global `asyncio.Semaphore`** across all datasets' images (move out of per-call `caption_images()`)
- **Callback routing** — route `CaptioningCallbacks` back to the right job/dataset
- **Error isolation** — one dataset's `BatchAbortedError` must not kill other datasets
- **Event emission** — maintain image→dataset mapping in a shared runner

Only worth it if the API provider charges per-connection or you need strict global rate limiting. Doesn't conflict with the queue approach — can be layered on top later.

## Key files

- `yadc/api/services/captioning/service.py` — global semaphore / queue
- `yadc/api/services/captioning/models.py` — `"queued"` JobStatus
- `yadc/api/services/captioning/job.py` — acquire semaphore in `_ado_run`
- `yadc/api/configuration.py` — `max_concurrent_jobs` config field
- `yadc/webui/src/lib/stores/events.ts` + `caption/status.ts` — handle `queued` status display (extend the `CaptioningStatusZ` schema and the `CaptioningStatus` enum)
