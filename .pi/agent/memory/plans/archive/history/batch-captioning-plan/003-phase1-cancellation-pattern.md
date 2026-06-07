---
date: 2026-06-06
---
# Phase 1 Cancellation: Eager Tasks + Manual Drain

**Context:** Plan said "use `asyncio.gather(*tasks, return_exceptions=True)`" but with `return_exceptions=True`, `CancelledError` is treated as a result rather than a trigger to cancel siblings. That would leak semaphores and in-flight HTTP requests on cancellation.

**Decision:** Use `asyncio.gather(*tasks)` (default `return_exceptions=False`) + manual sibling cancel in the `except` block. On any exception (CancelledError or BatchAbortedError), cancel all unfinished tasks and drain via `gather(return_exceptions=True)` before re-raising.

**Rationale:** Eager task creation (all images get `create_task` upfront) is simpler than lazy creation. The semaphore already throttles how many can be inside the model. Cancellation needs to release the in-flight HTTP requests to avoid leaking connections — the manual drain handles that.

**Files touched:** `yadc/core/captioning/runner.py` (`caption_images` task management).
