---
date: 2026-06-05
---
# Review Fixes: 400 for no api_url, orchestrator cleanup, plan doc sync

**Context:** Follow-up to `001-review-pass.md`. The user reviewed the
3 temporary commits and approved all fixes except the f-string-in-assert
item (#8), which they want to handle separately. This entry tracks the
fixes that were applied.

**Files touched:** `yadc/api/controllers/api_envs.py`,
`tests/api/test_envs.py`, `yadc/cmd/envs/models.py`,
`yadc/captioners/api/koboldcpp.py`, `yadc/captioners/api/gemini.py`,
`.pi/agent/memory/plans/list-models-captioner-reuse-plan.md`

## Fixes Applied

### 1. Restored 400 BAD_REQUEST for "no API URL configured" (must fix #1)

The orchestrator's `ValueError` covers two distinct cases (no api_url,
malformed upstream response), and the controller was mapping both to
`502 UPSTREAM_ERROR`. Misconfigured envs should be `400 BAD_REQUEST` —
the frontend can show "the env is misconfigured" vs "we couldn't reach
the API" differently.

**Change:** Short-circuit in the controller after the 404 check:

```python
if not env_data.api_url.value:
    return jsonify_error("Environment has no API URL configured", status=400, code=ErrorCode.BAD_REQUEST)
```

The orchestrator's `ValueError` is now reserved for the malformed-response
case.

**New test:** `test_returns_400_when_env_has_no_api_url` in
`tests/api/test_envs.py`. Verifies the controller returns 400 with
`code: BAD_REQUEST` and that `cmd_envs.list_models` is NOT called.

### 2. Refactored orchestrator to use `APICaptioner.create()` and the new instance method (should fix #4 + #5)

**Before:**
```python
inferred = await _infer_api_type_async(async_session, config.api.url)
captioner = APICaptioner(
    api_type=inferred, api_url=..., api_token=...,
    async_session=async_session, _warnings=False,
)
models = await captioner.inner_captioner.list_models(cache_ttl=cache_ttl)
```

**After:**
```python
captioner = await APICaptioner.create(
    api_url=..., api_token=...,
    async_session=async_session, _warnings=False,
)
models = await captioner.list_models(cache_ttl=cache_ttl)
```

Removed the now-unused `from yadc.captioners.api.api_captioner import
_infer_api_type_async` import — `APICaptioner.create()` wraps that call.

This also addresses the acknowledged #5 (instance method was dead code
from the orchestrator's perspective) — the orchestrator now exercises
the instance method in production. The test
`TestAPICaptionerListModels::test_delegates_to_inner_captioner` already
covered the delegate; the orchestrator refactor keeps that test green.

### 3. Added visibility for Koboldcpp's ignored `cache_ttl` (should fix #3)

Koboldcpp's `list_models` deliberately ignores `cache_ttl` because the
admin endpoint reflects on-disk state. The docstring explains this, but
a caller passing `cache_ttl=300` would silently get no caching. Added a
debug log when a non-None value is received:

```python
if cache_ttl is not None:
    _logger.debug("Koboldcpp list_models ignores cache_ttl=%s (admin endpoint is not cached).", cache_ttl)
```

### 4. Added inline comment to Gemini discovery path (acknowledged #7)

The fast path has an explicit `generateContent` capability check; the
discovery path relies on `_iter_generate_content_models()` filtering.
Added a one-line comment to make the implicit dependency obvious:

```python
# Capability filtering (``generateContent``) is handled implicitly
# by :meth:`_iter_generate_content_models` — unlike the fast path
# above, we don't need a separate ``generateContent`` check here.
```

### 5. Updated plan file to reflect the final design (must fix #2)

The plan was substantially out of sync with the implementation (4
documented decisions had drifted, phase wording was stale). Rewrote the
plan file in full:

- **Decisions section** now describes the final endpoint shape (dual
  GET/POST), method placement (instance method on captioners + module
  function in `cmd/envs/models.py`), constant location
  (`captioners/api/constants.py`), and orchestrator behavior.
- **Phase descriptions** updated to match what was actually built
  (notably Phase 3 is now "instance method delegate" not "classmethod
  orchestrator", and Phase 5 describes the real orchestrator content).
- **Files Touched** updated with the actual set of changed files
  (including tests).
- **Open Questions** section now lists the two outstanding TODO items
  tracked in `todo.md` instead of pretending everything is done.

## Test Counts

- Before fixes: 427
- After fixes: **428** (added `test_returns_400_when_env_has_no_api_url`)
- All passing
- Ruff clean
- Basedpyright: no new errors, only pre-existing warnings

## Items Deferred

- **#8 (f-string in assert)** — user wants to handle this separately.
  Noted in `001-review-pass.md` for context.
- **#6 (pre-existing inverted warning in `gemini._load_model`)** — not
  in scope of this refactor. Worth a follow-up TODO entry in `todo.md`
  if not already there. (Not added in this pass — user can decide.)
