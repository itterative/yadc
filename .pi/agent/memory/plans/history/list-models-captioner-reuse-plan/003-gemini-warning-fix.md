---
date: 2026-06-05
---
# Gemini: Fix Inverted Reasoning Warning + Apply to Both Paths

**Context:** The pre-existing bug flagged in
`001-review-pass.md` acknowledged #6. The warning at the end of
`GeminiCaptioner._load_model`'s discovery path had an inverted
condition: `if self._is_thinking_model and self._reasoning` — the
condition fires when the model **has** reasoning capabilities, but
the message says "**without** reasoning capabilities". On top of that,
the warning was only checked in the discovery path, so a non-thinking
model loaded via the fast path (direct `GET models/<name>`) with
`reasoning=True` would never trigger the warning.

User asked to fix the gemini issues. Initial attempt added 4
regression tests using `caplog`; user asked to drop the tests
(written into this entry for completeness), but kept the production
fixes.

**Files touched:** `yadc/captioners/api/gemini.py`

## Changes

### 1. Inverted condition fixed

**Before:**
```python
if self._is_thinking_model and self._reasoning:
    _logger.warning("Warning: selected a model without reasoning capabilities, but reasoning is enabled.")
```

**After (via the new `_warn_if_reasoning_unsupported` helper):**
```python
if self._reasoning and not self._is_thinking_model:
    _logger.warning("Warning: selected a model without reasoning capabilities, but reasoning is enabled.")
```

The condition now matches the message: warns when the user enabled
reasoning but the loaded model cannot support it.

### 2. Warning now fires from both paths

The warning was only checked after the discovery path. The fast
path set `self._is_thinking_model` and returned without the same
check, so the asymmetry was a latent bug. Extracted a
`_warn_if_reasoning_unsupported()` helper that's called from both
the fast path and the discovery path, after `self._is_thinking_model`
has been set. This ensures the warning fires regardless of which
path loaded the model.

### 3. Helper extraction

```python
def _warn_if_reasoning_unsupported(self) -> None:
    """Log a warning if the user enabled reasoning but the loaded
    model does not support thinking.

    Called by both the fast and discovery paths of :meth:`_load_model`
    after ``_is_thinking_model`` has been set, so the warning fires
    regardless of which path loaded the model.
    """
    if self._reasoning and not self._is_thinking_model:
        _logger.warning("Warning: selected a model without reasoning capabilities, but reasoning is enabled.")
```

Keeps the warning logic in one place. Cheap method (3 lines, no
state) and self-documenting via the docstring.

## Tests

User asked to drop the regression tests. The tests that were
written and then removed:

- `test_warns_when_reasoning_enabled_on_non_thinking_model_fast_path` — would have caught the original bug (warning never fired from fast path).
- `test_warns_when_reasoning_enabled_on_non_thinking_model_discovery_path` — sanity check for the discovery path.
- `test_no_warning_when_reasoning_disabled_on_non_thinking_model` — guards against false positives.
- `test_no_warning_when_thinking_model_used_with_reasoning` — would have caught the original inverted condition (it would fire a warning that should NOT fire).

All 4 used `caplog` to assert the presence/absence of the warning
at `logging.WARNING` on the `yadc.captioners.api.gemini` logger.

The production code is correct without these tests; the behavior
is now driven by a single helper, and the helper is small enough to
be obviously correct on inspection.

## Test Counts

- Before: 428
- After: 428 (no new tests — tests were written and dropped per user request)
- All passing
- Ruff clean

## Items Still Open

- **#8 (f-string in assert)** — user wants to handle this separately
- The other acknowledged issues from `001-review-pass.md` were either
  addressed in `002-review-fixes.md` or were out of scope
