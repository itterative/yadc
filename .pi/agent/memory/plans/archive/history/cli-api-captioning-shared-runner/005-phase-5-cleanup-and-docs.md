---
date: 2026-06-04
---
# Phase 5: Cleanup and architecture docs

**Context:** Phase 5 of the shared-runner plan was a no-op for code (the dead code was already removed in Phases 3 and 4) and a documentation pass.

**Decision:** No code changes — the dead code the plan called out was already gone:

- CLI's `_resolve_template` — removed in Phase 4 (the loader calls `resolve_template` internally).
- CLI's duplicate HTTP session / model creation blocks — removed in Phase 4 (replaced by `async with CaptioningRunner(...)`).
- API's `apply_config_overrides` / `resolve_template` re-exports — removed in Phase 3 (controllers and `services/__init__.py` import `CaptionJobOptions` directly from `yadc.core.captioning`).

**Documentation changes:**

- **New** `docs/captioning-runner.md` — canonical doc for the shared runner. Package layout, runner lifecycle, `CaptioningCallbacks` Protocol semantics, examples for both the CLI and API. Cross-references `captioning-workflow`.
- **Rewrote** `docs/captioning-workflow.md` — was CLI-specific (described the old `_load_dataset()` and inline captioning flow). Now covers the shared `load_dataset_config` + `CaptioningRunner` and notes what each side adds on top (CLI's interactive action menu, API's SSE event emission).
- **Updated** `docs/backend/core.md` — added the new `yadc/core/captioning/` package to the file tree and a `captioning-runner` cross-reference.
- **Updated** `docs/backend/cli.md` — `cli_caption.py` description now mentions the runner delegation.
- **Updated** `docs/backend/api.md` — `captioning.py` description now describes the runner delegation instead of the old in-line stream/save.
- **Updated** `docs/cli-cmd-structure.md` — the `caption` row's "pure logic" column is now `core/captioning/` (shared with the API) instead of "(uses captioners directly)". Added a one-line rationale.
- **Updated** `architecture-overview.md` — new `captioning-runner` doc added to the index.
- **Updated** `doc-management.md` — new doc added to the index with a one-line description.

**Files touched:** `.pi/agent/memory/docs/captioning-runner.md` (new), `.pi/agent/memory/docs/captioning-workflow.md` (rewritten), `.pi/agent/memory/docs/backend/core.md`, `.pi/agent/memory/docs/backend/cli.md`, `.pi/agent/memory/docs/backend/api.md`, `.pi/agent/memory/docs/cli-cmd-structure.md`, `.pi/agent/memory/architecture-overview.md`, `.pi/agent/memory/doc-management.md`.
