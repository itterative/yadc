---
date: 2026-06-06
---
# Phase 4: Tests Were Inlined into Phases 1–3

**Context:** The plan listed Phase 4 as a separate "Tests" phase with 3 test categories (runner, CLI, lock). Each phase was implemented with its own test file added as part of the change, so by Phase 3 there was nothing left to write.

**Decision:** Don't run a separate "Phase 4" pass. Update the plan to record the test coverage that already exists in `tests/core/captioning/test_runner.py::TestCaptionImages`, `tests/captioners/api/test_api_usage_lock.py`, `tests/cli/test_cli_caption.py`, and `tests/api/test_captioning_unit.py::TestAdoRunWithRunner`.

**Rationale:** Tests written alongside the code are more accurate (capture the design intent of the moment) and the diffs stay reviewable. A separate test pass after the fact tends to either be redundant (covering the same code again) or stale (testing the wrong thing because the code drifted).

**Files touched:** `plans/batch-captioning-plan.md` (this entry + Phase 4 marked done with the coverage list).
