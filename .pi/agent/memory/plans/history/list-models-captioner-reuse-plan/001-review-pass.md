---
date: 2026-06-05
---
# Review Pass: `list_models` Captioner Reuse (3 commits pending squash)

**Context:** User asked for a code review of the 3 temporary commits on the
`agent/frosty-cedar` branch (`0088ea9b tmp: save`, `1c4cca6d tmp: save gemini
fix`, `8d5e896a tmp: save password`). The commits refactor
`/api/envs/<name>/models` to reuse the existing `APICaptioner` system, add
a Gemini discovery-path regression fix, and add password-passing via dual
GET/POST. 427 tests pass. Ruff clean. Frontend svelte-check/eslint/prettier
clean. No regressions in the existing test suite.

**Note on commit count:** Branch summary said "4 temporary commits" but
only 3 exist (`0088ea9b`, `1c4cca6d`, `8d5e896a`). User was informed.

## Summary

End-to-end shape is sound. The split is right: `captioners/api` owns "how
to talk to a backend"; `cmd/envs` owns "which env to use and how to decrypt
its token"; the controller owns "HTTP shape + error mapping". The Gemini
discovery-path regression fix is a genuine bug fix. Tests are comprehensive
(41 new tests across 4 files). Frontend migration is clean. Squash of all
3 commits into a single `feat(api)` (with the Gemini fix folded in) is
recommended.

## Must Fix Before Squash

### 1. Error status code regression: "no API URL configured" returns 502 instead of 400

**Before** (deleted code): env exists but has no `api_url` → `400 BAD_REQUEST` with `"Environment has no API URL configured"`.

**After**: `cmd_envs.models.list_models` raises `ValueError("environment 'default' has no api_url configured")`, which the controller maps to `502 UPSTREAM_ERROR`. The orchestrator's `ValueError` covers two distinct cases (no api_url, malformed response) and the controller can't distinguish them.

**Fix:** Short-circuit in the controller before calling the orchestrator:

```python
env_data = cmd_envs.get_env(name)
if env_data is None:
    return jsonify_error("Environment not found", status=404, code=ErrorCode.NOT_FOUND)

if not env_data.api_url.value:
    return jsonify_error("Environment has no API URL configured", status=400, code=ErrorCode.BAD_REQUEST)
```

Add a controller test asserting 400 for the no-url case. The existing orchestrator test (`test_raises_when_env_has_no_api_url`) stays as-is — it asserts the `ValueError` at the orchestrator layer.

**Files touched:** `yadc/api/controllers/api_envs.py`, `tests/api/test_envs.py`

### 2. Plan documentation drift

The plan file `.pi/agent/memory/plans/list-models-captioner-reuse-plan.md` is out of sync with the final implementation:

- **Endpoint**: plan said "GET", implementation is "GET + POST" (with note about X-YADC-Password migration TODO).
- **Method placement**: plan said "`APICaptioner.list_models()` as a classmethod orchestrator", but the orchestrator was inlined into `cmd/envs/models.py` (no classmethod). The instance `APICaptioner.list_models()` is now a thin delegate.
- **Constants location**: plan said "`captioners/api/api_captioner.py` exports `DEFAULT_MODELS_CACHE_TTL_SECONDS`" — actual location is `captioners/api/constants.py` (the cycle-avoidance decision).
- **Phase 1 / Phase 3 wording is stale**: Phase 1 describes a classmethod orchestrator on `APICaptioner`; Phase 3 describes `cmd/envs/models.py` as a thin wrapper (the actual content is the orchestrator).

**Fix:** Either update the plan to reflect the final design (it's a "current state" doc) or move it to `archive/` since the implementation matches the conversation summary.

**Files touched:** `.pi/agent/memory/plans/list-models-captioner-reuse-plan.md`

## Should Fix

### 3. `KoboldcppCaptioner.list_models` silently ignores `cache_ttl` (footgun)

The override uses `cache_ttl: float | None = None` (base class default is `DEFAULT_MODELS_CACHE_TTL_SECONDS = 300.0`). The docstring explains why, but a caller passing `cache_ttl=300` will silently get no caching. Add a debug log when non-None:

```python
if cache_ttl is not None:
    _logger.debug("Koboldcpp list_models ignores cache_ttl=%s", cache_ttl)
```

**Files touched:** `yadc/captioners/api/koboldcpp.py`

### 4. Orchestrator could use `APICaptioner.create()` and the new `APICaptioner.list_models()` instance method (style + addresses #5)

```python
# current — duplicates APICaptioner.create + reaches through inner_captioner
inferred = await _infer_api_type_async(async_session, config.api.url)
captioner = APICaptioner(
    api_type=inferred, api_url=config.api.url, api_token=config.api.token,
    async_session=async_session, _warnings=False,
)
models = await captioner.inner_captioner.list_models(cache_ttl=cache_ttl)

# cleaner
captioner = await APICaptioner.create(
    api_url=config.api.url, api_token=config.api.token,
    async_session=async_session, _warnings=False,
)
models = await captioner.list_models(cache_ttl=cache_ttl)
```

**Files touched:** `yadc/cmd/envs/models.py`

## Acknowledged / No Action Needed

### 5. `APICaptioner.list_models()` instance method is only tested indirectly

If #4 is done, this is also fixed (orchestrator exercises the instance method in production). If #4 is skipped, the instance method is dead code from the perspective of the only current caller. Decide: keep + use (preferred), or remove (YAGNI).

### 6. `GeminiCaptioner._load_model` retains the pre-existing inverted warning

```python
if self._is_thinking_model and self._reasoning:
    _logger.warning("Warning: selected a model without reasoning capabilities, but reasoning is enabled.")
```

The condition fires when the model **has** reasoning capabilities, but the message says "**without** reasoning capabilities". Pre-existing bug, preserved through the refactor. **Not in scope** — but worth a follow-up TODO entry. (Not yet in `todo.md`; only the "decouple password" entry was added.)

### 7. `GeminiCaptioner._load_model` discovery path lacks explicit capability check comment

The fast path has `"model {model_repo} does not have generative capabilities: ..."` check; the discovery path relies on `_iter_generate_content_models()` filtering. Correct, but a one-line comment would help future readers. Optional.

### 8. `assert isinstance(model_resp_json, dict), f"..."` uses f-string in assert

Under `python -O`, `assert` statements are removed. The user-facing error is the `ValueError("failed to parse model list response")` raised on `pydantic.ValidationError`. The f-string is debug-only. **Fine, informational only.**

## Positive Observations

1. **Constants module breaks the import cycle cleanly** — `captioners/api/constants.py` is a leaf, imported by both `base.py` (for the abstract default) and `api_captioner.py` (for re-export).
2. **Method placement is consistent** — every backend gets the same `(self, cache_ttl)` signature. Inheritance handles `LLamacppCaptioner`/`OllamaCaptioner`/`VllmCaptioner`/`OpenRouterCaptioner` for free.
3. **The `_iter_generate_content_models` extraction in Gemini is well-designed** — yielding raw `GeminiModel` lets `list_models` and `_load_model` share pagination/filtering without coupling. The Gemini regression fix captures `matched_thinking` from the iterated model rather than rebuilding from names.
4. **`make_app_config_env` test helper** — was duplicated 3+ times in the old `test_envs.py`; the new helper with sensible defaults is a clear win.
5. **Frontend migration is clean** — the conditional `body.password` with `Record<string, unknown>` handles "no password set" and "session password set" cases without dead code.
6. **`Koboldcpp._load_model` refactor adds an early `break`** — the old code continued iterating after finding a match (wasteful since `available_models` is now a separate list). New code breaks. Small but correct.
7. **Test isolation is right** — per-backend `list_models` tests in `test_list_models.py`, orchestrator tests in `test_models.py`. Matches the architecture split.

## Squash Recommendation

3 commits split into:
1. `0088ea9b` — base + per-backend `list_models` + orchestrator + controller + tests
2. `1c4cca6d` — Gemini discovery path regression fix
3. `8d5e896a` — password-passing capability (dual GET/POST + frontend)

All three are tightly coupled and the Gemini fix is a true bug, not a separate feature. **Lean toward squashing all three** into a single `feat(api): reuse APICaptioner for /api/envs/<name>/models` (with the Gemini fix folded in as a normal part of the implementation, not a separate fix commit). Keeping them separate makes the Gemini fix easy to find but the cohesion isn't there.

## Test Counts

- Pre-refactor: 395
- After full refactor: 427 (32 new tests across 4 files)
- All passing
- New tests: `tests/cmd/envs/test_models.py` (7), `tests/captioners/api/test_list_models.py` (13), 2 Gemini regression tests in `test_gemini.py`, 10 new tests in `tests/api/test_envs.py` (incl. password forwarding, cache_ttl passthrough, GET-still-works regression)
