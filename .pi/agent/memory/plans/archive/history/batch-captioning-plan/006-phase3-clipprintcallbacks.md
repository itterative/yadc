---
date: 2026-06-06
---
# Phase 3: New `CLIPrintCallbacks` Class for Parallel Path

**Context:** The existing `CLICallbacks` class buffers per-token output for the no-stream case, then prints the buffer at the end. With multiple images in flight, the buffer is shared and tokens from different images interleave into a single buffer — the output becomes unreadable.

**Decision:** A new minimal `CLIPrintCallbacks` class for the non-interactive parallel path. `on_token` is a no-op (suppressed; tokens would interleave), `on_image_started` is a no-op, `on_image_captioned` logs the image path + duration, `on_image_error` logs the warning.

**Rationale:** In parallel mode the user is bulk-captioning; the per-token stream is not useful when it can't be attributed to a single image. The saved caption is on disk and can be reviewed there. Per-image completion timestamps give the user progress feedback.

**Trade-off:** The existing `CLICallbacks` is unchanged and still used for the legacy `_caption` path (sequential, dry-run + save per image with the action menu). Two callback classes is the price of keeping the legacy UX intact while supporting the new parallel path cleanly.

**Files touched:** `yadc/cli_caption.py` (new class), `tests/cli/test_cli_caption.py` (3 tests for the new class).
