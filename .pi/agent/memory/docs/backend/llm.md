---
name: backend/llm
description: Backend-agnostic LLM client layer under yadc/llm/ — BaseLLMClient, OpenAI-compatible + Gemini clients, per-server thin clients, create_client factory, Message/StreamChunk/MessageStream types. Composed by APICaptioner; self-contained (shared HTTP infra lives here too).
category: architecture
---

# Backend: LLM Client Layer (`yadc/llm/`)

A standalone client package for talking to LLM backends (OpenAI,
OpenRouter, Gemini, llama.cpp, KoboldCpp, vLLM, Ollama) without any
captioner / image / Jinja concerns. The prompt generator uses it
directly; `APICaptioner` composes it.

## Relation to captioners

`APICaptioner` **composes** a `BaseLLMClient` and delegates all
prediction to it. The client takes
ready-built `list[Message]` + a `reasoning` effort kwarg + `**opts`
(arbitrary body overrides); it never touches images, Jinja, or
`PredictionContext`. `assistant_prefill`, `extra_messages` (reply
history), structured `conversation_overrides` / advanced-settings, image
encoding, and Jinja prompt rendering all live on `APICaptioner` — they
build the `list[Message]` the client consumes.

The package is **self-contained**: the shared HTTP infra
(`async_session.py`, `cache.py`, `response_logger.py`,
`error_normalization.py`, `response_models.py`, `units.py`,
`constants.py`) lives in `yadc/llm/`. The
`captioners.api → llm` direction is the only one (no more bidirectional
package dependency). `normalize_error()` is consumed by both the client
layer and `APICaptioner` (for surface strings); `thinking.py` and the
captioner-side constants stay in `yadc/captioners/api/`.

## Critical: true streaming

The legacy `AsyncSession.request(stream=True)` is **fake streaming** —
it pops `stream` and never passes it to httpx, so the entire body is
buffered before any line is yielded. (Captioners still use this path,
unchanged.) The llm layer needs real token-by-token streaming (so
`MessageStream.cancel()` is meaningful), so `AsyncSession.open_stream()`
was added: built on `client.send(req, stream=True)`, returns an
`AsyncStreamHandle` (open response + idempotent `aclose` + debug-log
accumulation). Only `yadc/llm/` uses `open_stream` today.

## The `MessageStream` contract

`predict_next_message_stream` is `async def` (caller awaits). The await
performs the POST and opens the streaming response (preflight /
connection errors — including the friendly `ValueError("Connection
closed unexpectedly…")` for `httpx.RemoteProtocolError`/`ReadError` —
surface here). The returned `MessageStream` wraps the already-open
response:

- `async for chunk in stream` → `StreamChunk` (text / reasoning /
  reasoning_summary / reasoning_encrypted), auto-folded into `.message`.
- `await stream.collect()` → drains and returns the accumulated
  `Message` (fast path if already drained).
- `.message` → partial-so-far `Message`.
- `await stream.cancel()` → idempotent close of the underlying response.
- Cancellation also propagates via `CancelledError` raised in `__anext__`
  when the iterating task is cancelled.

Stream-only — no non-streaming method. Callers who don't care about
chunks use `await stream.collect()`.

## Per-server differences

All OpenAI-compatible clients share `OpenAICompatibleLLMClient`; the thin
subclasses only override `_customize_body` (and `load_model` where a
server has a bespoke loading endpoint):

- `openai.py` / `OpenAILLMClient` — identity only.
- `openrouter.py` — OpenRouter's own `reasoning` block
  (`{enabled, effort, exclude}`), `max_tokens`, `usage.include`, drops
  `stream_options`, treats `"[REDACTED]"` reasoning as redacted. Also
  overrides `load_model` to log remaining credits (`GET /credits`).
- `llamacpp.py` / `vllm.py` / `ollama.py` — `max_tokens` rename.
- `koboldcpp.py` — `max_tokens` + `<|im_end|>` stop token + bespoke
  `load_model` (`/api/admin/reload_config` + poll `/api/v1/model`).

`reasoning_exclude` (captioner's `reasoning_exclude_output`) flows through
`predict_next_message_stream` → `_build_body` as a body transient
(`_reasoning_exclude`) that OpenRouter consumes for `reasoning.exclude`;
Gemini reads it directly for `thinkingConfig.includeThoughts`.

`normalize_error()` is async — it `await`s `response.aread()` for
`httpx.HTTPStatusError` from a streaming response (whose body hasn't
been consumed; accessing `.text` directly would raise
`ResponseNotRead`). All call sites are inside async functions, so the
`await` is free.

`Message.role` is `str` (not a `Literal`) so the captioner can honour the
advanced `system_role` override (e.g. `"developer"`); the Gemini client
treats `system`/`developer` as the system message and maps `assistant`→`model`.

```
yadc/llm/
  __init__.py              # re-exports (+ __all__)
  types.py                 # Message, TextPart, ImageUrlPart, ImageUrl, ContentPart, StreamChunk, MessageStream, apply_chunk
  base.py                  # BaseLLMClient (owns AsyncSession/cache/logger/usage), APIUsage, default existence-check load_model
  openai_compatible.py     # OpenAICompatibleLLMClient (body building, SSE chunk decoding, reasoning translation)
  openai.py                # OpenAILLMClient (thin)
  openrouter.py            # OpenRouterLLMClient (reasoning fmt + max_tokens + usage + redacted detection)
  vllm.py / llamacpp.py / ollama.py   # thin (max_tokens rename)
  koboldcpp.py             # KoboldcppLLMClient (max_tokens + stop + bespoke load_model + admin list_models)
  gemini.py                # GeminiLLMClient (generateContent, thinkingConfig, paginated models, thought:true reasoning)
  factory.py               # create_client(*, api_url, api_token, cache=, response_logger=, async_session=, **overrides) + _infer_api_type() (URL + probe routing → client class). Takes primitives so both the env UserConfig path and the dataset Config path (runner) work without coupling to either config model.
  constants.py             # DEFAULT_MODELS_CACHE_TTL_SECONDS (LLM-side; DEFAULT_LIST_MODELS_TIMEOUT_SECONDS stays in yadc/captioners/api/constants.py as it's orchestrator-side)
  cache.py                 # HTTPResponseCache (file-based JSON, sha256-keyed)
  response_logger.py       # ResponseLogger (JSONL debug log; sanitises auth headers)
  response_models.py       # Pydantic response models (OpenAI/Gemini/OpenRouter/Kobold)
  error_normalization.py   # normalize_error() + GenerationError (shared with captioner layer for surface string)
  units.py                 # size_units() pretty-print
  async_session.py         # AsyncSession (retries, cache, open_stream, capture_response) + AsyncStreamHandle
```

## See also

- `backend/captioners` — the captioner layer that composes a client from this package.
- `captioner-architecture` — the captioner hierarchy + mixin pattern + image limits.
- `.pi/agent/memory/plans/prompt-generator-plan.md` — the plan that motivated the package and the captioner collapse; Phases 1a + 1b are done.
