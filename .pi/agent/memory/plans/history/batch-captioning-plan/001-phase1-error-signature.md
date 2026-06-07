---
date: 2026-06-06
---
# Phase 1 Error Signature: HTTP Code Per Signature

**Context:** The plan said "identical errors" without specifying the granularity. Naive implementation matched the regex pattern string (e.g. `"^api returned an error \(http \d+\)"`), so any 4xx — 401, 404, 422 — counted as the same signature and accumulated toward the abort threshold.

**Decision:** The signature includes the captured group (the HTTP code), so 401 and 404 are distinct signatures. The mixed-errors test (`API, API, image, API, API, image` not aborting) and the changing-signature test (`2x 401, 2x 404` not aborting) both pass.

**Rationale:** A burst of 404s after a burst of 401s is a real signal that the problem is changing (likely a misconfiguration caught and partially fixed) — not a persistent single-mode failure. The original would have aborted the batch on a transient signature change, which would be a false positive.

**Files touched:** `yadc/core/captioning/runner.py` (signature format strings), `tests/core/captioning/test_runner.py` (new test classes).
