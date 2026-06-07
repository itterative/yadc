---
date: 2026-06-06
---
# Phase 1 Image-Error Reset: Image Errors Reset the API-Error Counter

**Context:** First implementation only reset the counter on success or on signature change. A 401 → 401 → image-error → 401 → 401 sequence would hit the threshold of 3 and abort, even though the image error in the middle should have broken the chain. The test `test_mixed_errors_dont_trigger_abort` caught this.

**Decision:** Image-attributable errors (no signature match) reset both the counter and the last signature. The 401 → 401 → image → 401 → 401 sequence now keeps the counter at 0/1/0/1/0 and never aborts.

**Rationale:** An image error between two API errors is a different kind of failure (this specific image is bad) than two identical API errors (the API is broken). They shouldn't be conflated.

**Files touched:** `yadc/core/captioning/runner.py` (`_BatchErrorTracker.record_error` early-return path).
