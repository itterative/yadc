---
date: 2026-06-06
---
# Phase 3: Interactive + Parallel Deferred (User-Confirmed)

**Context:** The plan said "decision pending review of the work involved" for the interactive + `--max-concurrent > 1` case. Two options were on the table:
1. Error out and tell the user to use `--non-interactive`
2. Build a queue that streams N captions concurrently but presents them one at a time for review

**Decision:** Option 1 (error out). User confirmed: "interactive mode should be incompatible with concurrent captioning. even if we do caption let's say a buffer of images, we'd still need to approve one by one, so I wouldn't do this yet. you may leave it there as it is for now in terms of the plan."

**Rationale:** The action menu is inherently per-image (continue / retry / reply / edit). A buffered parallel review would need to either (a) pause the model to let the user review each caption (defeating parallelism), or (b) auto-accept everything that arrives while the user is reviewing (defeating interactive mode). Both defeat the purpose. Defer until someone has a concrete need.

**Files touched:** `yadc/cli_caption.py` (early check + `CLIPrintCallbacks`), `tests/cli/test_cli_caption.py` (new).
