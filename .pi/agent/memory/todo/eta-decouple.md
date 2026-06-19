---
name: eta-decouple
description: Decouple captioning ETA from the captioningStatuses store by tracking job identity separately
category: frontend
priority: 2
---

# Decouple ETA from the captioningStatuses store

## Problem

The ETA estimation in `DatasetTopbar` is coupled to the per-dataset status entry carrying `api_url`/`api_model_name`. `_estimateRemainingSeconds` reads these from the status to look up the timing ring buffer.

This means:
- ETA only works when the status store happens to have API info
- `caption/actions.ts` must seed `api_url`/`api_model_name` from the HTTP response into the status store, mixing progress concerns with job identity concerns
- Single-image captioning was broken until we started seeding API info (commit 2e13d365)

## Current flow

1. `startBatchCaptioning` / `captionSingleImage` seed the status entry with `api_url` + `api_model_name` from HTTP response
2. SSE events also update the status entry with API info
3. `DatasetTopbar` reads `captionStatus.api_url` and `captionStatus.api_model_name` (from its `captionStatus` prop) for ETA lookup

## Better design

Track a separate `_currentCaptioningJob` store (or similar) with `{ api_url, api_model_name, job_id }`:
- Set it when a job starts (from HTTP response)
- Update it from SSE events
- Clear it when job reaches terminal state (`done`/`error`/`cancelled`/`idle`)

Then `DatasetTopbar` ETA uses the job identity store for ring-buffer lookup, and the status entries only carry progress (`status`, `processed`, `total`, `errors`).

## Files involved

- `src/lib/stores/caption/jobs.ts` — add a new store for job identity `{ api_url, api_model_name, job_id }` + set/clear lifecycle
- `src/lib/stores/events.ts` — update the `captioning_status` handler to also update the job-identity store
- `src/lib/stores/caption/actions.ts` — stop seeding `api_url`/`api_model_name` into `captioningStatuses`; seed the job-identity store instead
- `src/routes/datasets/[name]/DatasetTopbar.svelte` — use the job-identity store for ETA lookup instead of the `captionStatus` prop
