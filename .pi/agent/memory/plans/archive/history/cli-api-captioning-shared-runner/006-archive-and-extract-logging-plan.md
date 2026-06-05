---
date: 2026-06-04
---
# Plan archived; logging refactor extracted to its own plan

**Context:** Phases 1–5 of this plan are complete. Phase 6 (logging unification) was always a separate workstream that was sketched in the runner plan because there was no other natural home for it.

**Decision:** Extract Phase 6 into its own plan (`plans/cli-api-logging-unification-plan.md`) and archive this plan. The runner plan was the foundation for the logging refactor — once the runner landed, the right level of detail for the logging work became clear enough to live in its own document.

**Files touched:**
- `plans/cli-api-captioning-shared-runner-plan.md` → moved to `plans/archive/cli-api-captioning-shared-runner-plan.md`
- `plans/history/cli-api-captioning-shared-runner/` → moved to `plans/archive/history/cli-api-captioning-shared-runner/`
- `plans/cli-api-logging-unification-plan.md` (new) — extracted Phase 6 from this plan, expanded to its own phase breakdown (refactor core / rewrite API wrapper / cleanup + docs)
- `plan-management.md` — index updated: this plan removed; new logging plan added
