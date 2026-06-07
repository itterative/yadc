---
date: 2026-06-06
---
# Phase 2: Wire Field Name — `max_concurrent`, Not `batch_size`

**Context:** The plan said "The `batch_size` is already a `CaptionJobOptions` field" but the actual field on `CaptionJobOptions` (set in Phase 1) is `max_concurrent`. First implementation of the frontend used `batch_size` on the wire, which is silently dropped by `CaptionJobOptions(extra="ignore")` — no error, but the parallelism knob doesn't work. Caught by an ad-hoc spot check that round-trips the assembled options through `CaptionJobOptions.model_validate`.

**Decision:** Frontend sends `max_concurrent` to match the backend. User-facing label is "Concurrent requests"; store key is `batchSize`; wire field is `max_concurrent`. Documented in the plan as a "field-name gotcha" so future contributors don't make the same mistake.

**Rationale:** The runner's `asyncio.Semaphore` is fundamentally about "max concurrent in-flight", not "batch size" (a batch of N could be processed one at a time). `max_concurrent` is the precise name and the same one the backend uses. Renaming the backend field would be churn for no benefit.

**Files touched:** `yadc/webui/src/lib/components/caption/CaptionSettingsPanel.svelte`, `yadc/webui/src/lib/stores/caption/options.ts`.
