---
date: 2026-06-28
---
# Plan-body prune

**Context:** With Phases A/B/C complete, the plan body carried
implementation detail and design rationale that duplicated the
architecture reference doc (`docs/tagger-architecture.md`) — verbose
phase breakdowns (A1–A5, B1–B4), an "Open decisions" section whose
questions were all resolved, and a "Notes / Decisions" block restating
the why-subprocess / why-lazy-spawn / why-per-category-thresholds
rationale. Per the memory convention (plans track phase/status history;
reference docs describe current architecture), this pruned the plan
to status + phase pointers and moved the detail out.

**Decision:** Slimmed the plan to a Status section, a condensed
`## Completed` list (one paragraph per phase, pointing at the matching
history entry for detail), `## Pending` (future iterations), and a short
`## Notes / Decisions` pointer to where the rationale now lives. Bumped
`last_history` 1 → 3.

**What moved, and where:**

| Pruned from plan | Now lives in |
|------------------|--------------|
| Phase A detail (A1 run endpoints, A2 save options + formatters, A3 backend write helpers, A4 interactive save endpoint, A5 job service) | `history/tagger-plan/001-webui-batch-phase.md` (decision-time); current API shape in `tagger-architecture.md` ("Save path + batch job", endpoint list) |
| Phase B detail (B1 stores, B2 interactive Tags tab, B3 batch side panel, B4 tests/verification) | `001-webui-batch-phase.md`; current frontend layout in `tagger-architecture.md` ("WebUI surface") |
| "Open decisions" (single-image sync vs job; batch panel placement; mobile density) | All resolved → folded into `001-webui-batch-phase.md` as the decision record (synchronous preview kept; new side-panel tab; 4th tab accepted) |
| "Notes / Decisions" rationale: why subprocess; why lazy spawn + idle teardown; why per-category thresholds; why not `pandas`; why CPU-only `onnxruntime` default + `[gpu]` extra | `tagger-architecture.md` (Lifecycle, Concurrency, threshold config rows, Model source / deps) — these describe the *current* architecture, so the reference doc is their home |
| Configuration field enumeration (each threshold, the idle timeout, the liveness knobs) | `tagger-architecture.md` config table (single source of truth — the plan was drifting out of sync as knobs were added) |
| `Completed` items 1–11 prose (scaffold, ONNX preprocessing steps, wire-protocol description) | `tagger-architecture.md` (type table, "Multiprocessing protocol", `OnnxTagger` section) |

**Rationale:** The plan was duplicating the architecture doc and
drifting (e.g., it still described `DELETE /tag` as plain "graceful
stop" after cancel shipped; it enumerated config fields that gained new
siblings). A slim plan that points at history (for decisions) and the
reference doc (for current shape) avoids the drift and keeps the plan
honest about what it's for: phase/status tracking. Reference docs win
for "what the system is today."

**Files touched:** `plans/tagger-plan.md` (rewritten), this entry.
