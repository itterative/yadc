---
date: 2026-06-04
---
# Phase 1: Copy instead of move

**Context:** The plan originally said to *move* `CaptionJobOptions`, `apply_config_overrides`, and `resolve_template` from `yadc/api/services/captioning.py` to the new `yadc/core/captioning/` package, with re-exports from the API module for backward compat.

**Decision:** In Phase 1, *copy* the code instead. Leave the API's existing copies untouched. The new `yadc/core/captioning/` package is strictly additive — no existing import paths change, no internal usage is affected.

**Rationale:** Two classes of risk were removed by copying:
1. `AsyncCaptionJob._apply_overrides` and `AsyncCaptionJob._resolve_template` use the local module-level functions. A re-export from the API would resolve correctly, but the diff is larger and touches the API in a way that's harder to revert if something goes wrong.
2. `CaptionJobOptions` would be defined in two places; `isinstance` checks or any code that depends on class identity would need to be careful. Copying keeps the two classes fully independent until Phase 3 explicitly migrates the API.

Future phases handle the deletion: Phase 3 (refactor `AsyncCaptionJob` to use the runner) will also switch the API to import `CaptionJobOptions` from the new location and delete the local definition. Phase 4 (refactor `cli_caption.py`) does the same for the CLI's inline logic. Until those phases land, the two copies coexist.

**Files touched:** `.pi/agent/memory/plans/cli-api-captioning-shared-runner-plan.md` (Phase 1 description and "files touched" updated), `yadc/core/captioning/{__init__,options,loader}.py` (new, copied from API).
