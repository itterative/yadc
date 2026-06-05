---
name: list-models-captioner-reuse-plan
description: Replace hand-rolled `/api/envs/<name>/models` HTTP/parsing with structured calls into the existing `APICaptioner` system. Add a shared `list_models()` method to each backend captioner, a `cmd/envs` wrapper, and a configurable cache TTL.
last_history: 3
---

# `list_models` Captioner Reuse — Plan

## Status

Implemented (pending squash + plan doc update)

## Problem

`yadc/api/controllers/api_envs.py::list_models` (the old code) manually
proxied `GET /models` to the env's `api_url`, using bare `requests`,
hand-rolled token decryption, and inline shape detection for OpenAI /
Ollama / fallback responses. None of this knowledge was shared with the
captioner code in `yadc/captioners/api/`, which already:

- Detects API type from the URL (`_infer_api_type_async`)
- Uses `AsyncSession` for retries, timeouts, caching, response logging
- Parses `OpenAIModelsResponse` / `GeminiModelsResponse` via Pydantic
- Hits per-backend endpoints (e.g. Koboldcpp's `/api/admin/list_options`)
- Caches model lists

## Goal

Make the captioners the single source of truth for "how to list models",
and make the API endpoint a thin HTTP wrapper around it.

## Decisions (final)

- **Endpoint shape**: `GET` or `POST /api/envs/<name>/models`. The dual
  method is the awkward compromise needed to support password-passing in
  the body without a header. `GET` is for the no-password / YADC_PASSWORD
  case. `POST` accepts `{"password": "..."}` matching the
  `reveal_env_value` / `set_key_mode` body shape.
- **Method placement**: `list_models()` is an **async instance method**
  on each `BaseAPICaptioner` subclass. `APICaptioner.list_models()` is
  a thin instance-method delegate to `inner_captioner.list_models()`. The
  orchestrator that ties env-resolution to captioner dispatch lives in
  `yadc/cmd/envs/models.py` as a module-level function (not a classmethod
  on `APICaptioner` — that would shadow the instance method).
- **Cache TTL**: 5 minutes by default (`300.0s`), configurable via
  `Configuration.api_models_cache_ttl`. The CLI captioner
  (`_load_model`) and the API endpoint both use the same constant.
  Koboldcpp's admin endpoint is the one exception — it reflects on-disk
  state and is never cached (the `cache_ttl` parameter is logged and
  ignored).
- **Where the constant lives**: `captioners/api/constants.py` (a leaf
  module). Imported by both `base.py` (for the abstract default) and
  `api_captioner.py` (for re-export). Putting it in either of those would
  create an import cycle.
- **Orchestrator session management**: `cmd/envs/models.py::list_models`
  builds a fresh `AsyncSession` if none is provided (caller can inject
  for testing). Uses `APICaptioner.create()` to combine type inference
  with construction. Delegates the actual `list_models` call to the
  new `APICaptioner.list_models()` instance method.
- **Error mapping in the controller**: `PasswordRequiredError → 403
  PASSWORD_REQUIRED`. `httpx.ConnectError` / `httpx.TimeoutException` →
  `502 UPSTREAM_ERROR`. `httpx.HTTPStatusError` → `502 UPSTREAM_ERROR`
  (with the upstream status code in the message). `ValueError` from the
  orchestrator → `502 UPSTREAM_ERROR` (malformed response). A
  misconfigured env (no `api_url`) short-circuits with `400
  BAD_REQUEST` *before* calling the orchestrator — distinguishing a
  client-fix problem from an upstream error.
- **Frontend**: `_fetchModels` sends `POST` with the session password
  in the body if `sessionPassword` is set. The endpoint also accepts
  `GET` for callers that can't send a body.
- **Pending TODO** (deferred): migrate all three password-passing
  endpoints (`/reveal`, `/key-mode`, `/models`) to a single
  `X-YADC-Password` header. See `todo.md` for the migration plan.

## Implementation Phases

### Phase 1 — `list_models()` on each backend captioner

`yadc/captioners/api/base.py`:
- Add abstract `async def list_models(self, cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS) -> list[str]: ...`.

`yadc/captioners/api/openai.py`:
- Implement `OpenAICaptioner.list_models()`: `GET models` →
  `OpenAIModelsResponse` → `[model.id for model in models.data]`.
  Cache via `cache_ttl` (default constant).
- Refactor `_load_model()` to call `self.list_models()` internally (no
  duplication).

`yadc/captioners/api/llamacpp.py`, `ollama.py`, `vllm.py`,
`openrouter.py`: no override needed (they inherit from
`OpenAICaptioner`).

`yadc/captioners/api/gemini.py`:
- Implement `GeminiCaptioner.list_models()`: paginated `GET models` →
  filter to those with `generateContent` → strip `models/` prefix.
- Extract `_iter_generate_content_models()` helper yielding raw
  `GeminiModel` objects, shared by `list_models()` (names only) and
  `_load_model()` (needs `thinking` flag). **Critical**: the discovery
  path of `_load_model()` must capture the matched model's
  `model.thinking` flag — a refactor that routed through
  `list_models()` (which only collects names) silently dropped the
  flag. See `history/001-review-pass.md` for the regression
  detection.
- Refactor `_load_model()` to use the helper for both fast and
  discovery paths.

`yadc/captioners/api/koboldcpp.py`:
- Implement `KoboldcppCaptioner.list_models()`: `GET
  /api/admin/list_options` → `KoboldAdminSettingsReponse` → list of
  strings, skipping `"unload_model"`. The `cache_ttl` parameter is
  ignored (admin endpoint reflects on-disk state).
- Refactor `_load_model()` to reuse it. Added an early `break` after
  match (the old code continued iterating wastefully).

### Phase 2 — Constants + re-exports

`yadc/captioners/api/constants.py` (new): holds
`DEFAULT_MODELS_CACHE_TTL_SECONDS = 300.0`.

`yadc/captioners/api/__init__.py`: re-export
`DEFAULT_MODELS_CACHE_TTL_SECONDS`.

### Phase 3 — `APICaptioner` instance method

`yadc/captioners/api/api_captioner.py`:
- Add `APICaptioner.list_models()` instance method that delegates to
  `self.inner_captioner.list_models(cache_ttl=cache_ttl)`.

### Phase 4 — Configuration

`yadc/api/configuration.py`:
- Add `api_models_cache_ttl: float = DEFAULT_MODELS_CACHE_TTL_SECONDS`.

### Phase 5 — `cmd/envs` orchestrator

`yadc/cmd/envs/models.py` (new): `list_models(env, *, password, cache,
cache_ttl, async_session) -> list[str]`. Loads the env via
`load_env`, builds an `AsyncSession` (or uses the injected one),
infers the API type via `APICaptioner.create()`, calls
`captioner.list_models()`, returns a sorted list.

`yadc/cmd/envs/__init__.py`: re-export `list_models`.

### Phase 6 — Controller refactor

`yadc/api/controllers/api_envs.py`:
- Route: `@app.route("/envs/<name>/models", methods=["GET", "POST"])`.
- 404 if env not found. **400 if env has no `api_url`** (short-circuit
  before calling the orchestrator).
- POST reads `{"password": "..."}` from the body. GET doesn't.
- Maps `cmd_envs.list_models` exceptions to HTTP status codes (see
  "Error mapping" above).
- Response shape: `{"models": [...], "default": "<api_model_name>"}`
  (the `default` field is omitted when `api_model_name` is unset).

### Phase 7 — Frontend

`yadc/webui/src/lib/stores/env/api.ts`:
- `_fetchModels` now uses `POST` with `Content-Type: application/json`.
- Body includes `sessionPassword` if set.

### Phase 8 — Tests

- `tests/captioners/api/test_list_models.py` (new): per-backend
  coverage — OpenAI family (parametrized over all 5 backends), Gemini
  (filtering + pagination), Koboldcpp (skip unload_model),
  `APICaptioner.list_models()` delegate.
- `tests/cmd/envs/test_models.py` (new): orchestrator coverage —
  returns sorted models, raises on no api_url, forwards password and
  cache_ttl, propagates upstream errors.
- `tests/captioners/api/test_gemini.py`: 2 new regression tests for
  the `_is_thinking_model` discovery-path capture (one with
  `thinking=True`, one with `thinking=False`).
- `tests/api/test_envs.py`: 10 new tests for the controller — 404,
  400 (no api_url), 403 (password required), 502 (connection error,
  HTTP error, value error), happy path, default field omitted, cache
  TTL passthrough, password forwarding, GET still works. Plus a
  `make_app_config_env` helper with sensible defaults.

## Files Touched

- `yadc/captioners/api/api_captioner.py` (instance method delegate)
- `yadc/captioners/api/base.py` (abstract method)
- `yadc/captioners/api/openai.py` (implement + refactor `_load_model`)
- `yadc/captioners/api/gemini.py` (implement + refactor `_load_model`
  + helper extraction)
- `yadc/captioners/api/koboldcpp.py` (implement + refactor `_load_model`)
- `yadc/captioners/api/constants.py` (new)
- `yadc/captioners/api/__init__.py` (re-export)
- `yadc/api/configuration.py` (add `api_models_cache_ttl`)
- `yadc/cmd/envs/models.py` (new — orchestrator)
- `yadc/cmd/envs/__init__.py` (re-export)
- `yadc/api/controllers/api_envs.py` (refactor + dual-method route)
- `yadc/webui/src/lib/stores/env/api.ts` (POST with password)
- `tests/captioners/api/test_list_models.py` (new)
- `tests/cmd/envs/test_models.py` (new)
- `tests/captioners/api/test_gemini.py` (2 new regression tests)
- `tests/api/test_envs.py` (10 new tests + helper)

## Open Questions

None — the implementation is complete pending user review. The two
outstanding TODO items are tracked in `todo.md`:
1. Migrate all password-passing endpoints to `X-YADC-Password` header
2. Fix the pre-existing inverted warning in `gemini._load_model`
