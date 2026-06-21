---
date: 2026-06-20
---
# Phase 2 — Backend prompt generation (complete)

**Context:** The prompt-generator plan's Phase 2 — the backend
plumbing for generating Jinja2 prompt templates via any configured
env's model.

**Decision:** Implemented as designed, then **reworked after
review** to add multimodal example support:

- `yadc/api/services/prompt_generation.py` — `PromptGenerationService`
  + `ExamplePair` + `PromptGenerationRequest` Pydantic models.
  Builds the LLM client directly via `yadc.llm.create_client` (no
  captioner — no image involved), calls `load_model`, then yields
  `StreamChunk`s from `predict_next_message_stream(messages,
  reasoning=None)`.
- `yadc/api/controllers/api_prompts.py` — `POST /api/prompts/generate`
  returns a streaming NDJSON response (one JSON object per line:
  `{"type": "token", "text": "...", "reasoning": "..."}`,
  `{"type": "done"}`, or `{"type": "error", "message": "..."}`).
  Preflight errors (password required, missing `api_url`,
  missing `api_model_name`) return the standard JSON error responses
  *before* the stream starts so the HTTP status code can be 403/400.
  Mirrors `api_captioning.start_captioning`'s preflight pattern.
- Tests in `tests/api/test_prompt_generation.py` (19 tests, service-
  level with mocked `BaseLLMClient`) and `tests/api/test_api_prompts.py`
  (18 tests, controller-level via Quart test client).

**Rework after initial review (multimodal examples):**

The initial `ExamplePair` had only `subject` + `caption` + a
`source` discriminator (`"dataset"` vs `"manual"`). After review it
became clear that examples should carry images so the LLM can
mirror the example's visual style, not just its caption text.

Changes:
- **Dropped `source`** from `ExamplePair`. The frontend now handles
  *all* image acquisition — manual uploads are read and
  base64-encoded in the browser; dataset examples are fetched via
  the existing `GET /datasets/<name>/images/<id>/image` endpoint
  and base64-encoded by the frontend. The backend stays stateless
  about image sources.
- **`image_data_url` is required** on every example (`str`, format
  `data:<mime>;base64,...`).
- **Multi-turn meta-conversation** with priming assistant turns.
  The conversation is structured so the LLM acknowledges each
  example briefly and waits for a final "now generate" user turn:

      User:      intent + "I'll send N examples, then ask you to generate"
      Assistant: "Understood. Send your examples."          (fake priming)
      User:      example 1 (text + image parts)
      Assistant: "Got it. Send the next example."            (fake priming, omitted after last)
      User:      example 2
      ...
      User:      example N
      User:      "Now generate the Jinja2 template."
      Assistant: <real LLM response>

  The 0-examples path skips priming and just sends the intent +
  "now generate" in a single user turn.
- The frontend implication (full file fetch + base64-encode for
  dataset images) is Phase 3 work.

**Design choices made during implementation:**

- **Preflight split**: Controller does a synchronous `cmd_envs.load_env`
  preflight (same as `api_captioning`) so password-required surfaces
  as a real 403 instead of a stream-level error line. The service
  still does its own preflight for direct callers (tests, future CLI).
- **Streaming controller pattern**: Mirrors the upload streaming
  endpoints in `api_datasets.py` — `async def _stream(): ...` returns
  `yield json.dumps(...) + "\n"` lines, wrapped in
  `Response(_stream(), mimetype="application/x-ndjson")` with
  `Cache-Control: no-cache` + `X-Accel-Buffering: no`.
- **Meta-prompt lives in the service module**: `_SYSTEM_PROMPT`
  module-level constant in `prompt_generation.py`. Phase 5 of the plan
  is explicitly for tuning.
- **`reasoning=None` always**: The artifact is the template body
  itself; reasoning wastes tokens. Phase 5 may revisit.
- **Service `aclose` in finally**: Wrapped in try/finally so the
  client + its session are closed on natural exhaustion, caller
  cancel (via Quart's response cancellation propagating
  CancelledError into the async generator), and stream errors.

**Files touched:**

- New: `yadc/api/services/prompt_generation.py`,
  `yadc/api/controllers/api_prompts.py`,
  `tests/api/test_prompt_generation.py`,
  `tests/api/test_api_prompts.py`
- Updated: `yadc/api/services/__init__.py` (re-export new types),
  `.pi/agent/memory/docs/backend/api.md` (new controller + service
  listed in the directory tree)

**Open Questions status:**

- **Cancel propagation (end-to-end)**: Frontend `AbortController`
  + `fetch.abort()` chain not exercised yet — that's Phase 3 work.
  The server-side cancellation path is in place: Quart cancels the
  response coroutine on client disconnect, which propagates
  `CancelledError` into the async generator's `async for`, which
  closes the underlying `MessageStream` and the upstream HTTP
  response. Tests verify the cancellation path indirectly (the
  service's `client.aclose()` runs in `finally`).
