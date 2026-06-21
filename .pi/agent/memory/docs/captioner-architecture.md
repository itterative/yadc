---
name: captioner-architecture
description: Captioner hierarchy — APICaptioner auto-detection, inner captioner delegation, mixin pattern, and per-backend details.
category: architecture
keep_updated: true
---

# Captioner Architecture

## Hierarchy

```
Captioner (abc)                        # core/captioner.py
  ├─ load_model(), unload_model(), offload_model()
  ├─ predict(), predict_stream()
  ├─ prompts_from_image()              # Jinja2 template rendering
  └─ _encode_image()                   # base64 with auto-resize/quality degradation

APICaptioner(Captioner, ThinkingMixin)  # captioners/api/api_captioner.py
  └─ Composes a BaseLLMClient (yadc/llm/) — the only API captioner class left
  └─ Owns captioning glue: prompts, image encode, conversation_overrides,
     assistant_prefill, extra_messages, ThinkingMixin, PredictionContext
  └─ load_model/unload_model/offload_model/log_usage/list_models proxy to the client
  └─ predict()/predict_stream() build list[Message], call client.predict_next_message_stream,
     adapt StreamChunk → token stream (strip inline <think>, fold native reasoning into ctx)
```

The per-backend wire-format logic lives on the LLM clients in `yadc/llm/`
(see `backend/llm`). Construction: `client = await create_client(...)`
then `APICaptioner(client=client, ...)`.

## API Type Inference (`_infer_api_type()` in `yadc/llm/factory.py`)

`create_client(api_url, api_token, ...)` calls the module-level
`_infer_api_type(session, api_url)` once, then constructs the matching
`BaseLLMClient` subclass:

1. **URL domain check**: `api.openai.com` → OPENAI, `openrouter.ai` → OPENROUTER, `generativelanguage.googleapis.com` → GEMINI, `*-aiplatform.googleapis.com` → GEMINI
2. **Models endpoint**: `owned_by` field — "llamacpp", "koboldcpp", "vllm"
3. **Health/service endpoints**: `/health` header for llama.cpp, `/.well-known/serviceinfo` for koboldcpp, `HEAD /` for Ollama, `/api/version` for Ollama, `/ping`+`/version` for vLLM
4. **Fallback**: OPENAI

## Per-Backend Image Limits

| Backend | max_image_size | max_encoded_size |
|---------|---------------|-----------------|
| OpenAI | 1024×1024 | 10 MiB |
| OpenRouter | 1024×1024 | 10 MiB |
| Local (llamacpp, etc.) | 1536×1536 | 25 MiB |
| Gemini | 2048×2048 | 5 MiB |

## Model-List Timeout

`cmd_envs.list_models()` (and the controller's `/api/envs/<name>/models` endpoint) bounds the entire operation — API-type inference probes + the per-backend list call — with `asyncio.wait_for` using `Configuration.list_models_timeout` (default `DEFAULT_LIST_MODELS_TIMEOUT_SECONDS = 10.0`, in `captioners/api/constants.py`). On timeout the controller returns HTTP 504 `GATEWAY_TIMEOUT` so a dead/slow env can't leave the WebUI model picker spinning forever. `timeout=None` disables the cap (used by CLI scripts that don't need a hard bound).

## Mixins and Helpers

- **ThinkingMixin** (mixin, on `APICaptioner`): `_handle_thinking(content)` / `_handle_thinking_streaming_async(stream)` — strips `<think>...</think>` blocks from output, logs thinking content with `> ` prefix. Only fires for *inline* reasoning (llama.cpp `<think>` embedded in content); *native* reasoning (OpenAI `reasoning_details`, Gemini `thought:true`) is already separated by the client into `chunk.reasoning`, which `APICaptioner` folds straight into the `PredictionContext` — so it never reaches the visible caption and isn't routed through ThinkingMixin.
- **`normalize_error(error)`** (module-level helper, `yadc/llm/error_normalization.py`): shared by `APICaptioner` and the `yadc/llm/` clients. Not a mixin.

## Conversation Building

`APICaptioner._build_messages(image)` renders prompts + encodes the
image into `list[Message]`
(system + user[image] + reply history + optional assistant prefill). The
client translates to each backend's wire shape (OpenAI `chat/completions`
with `image_url` parts; Gemini `system_instruction` + `contents` +
`generationConfig`). Advanced-settings `conversation_overrides` (role
overrides, `assistant_prefill`, `extra_messages`, arbitrary body keys,
Gemini `safety_settings`/`generation_config`) are parsed here and flow to
the client as messages + `**opts`.

## PredictionContext

Passed via `prediction_context` kwarg. Populated by captioners during prediction:
- `reasoning` — raw thinking text
- `reasoning_summary` — displayable summary
- `reasoning_encrypted` — opaque data to pass back in subsequent turns (OpenAI reasoning_details)

## Extra Messages / Reply History

Multi-turn via `extra_messages: list[ReplyRound]`. Each ReplyRound has role (user/assistant), content, optional reasoning and reasoning_encrypted. The caption command builds reply history interactively.

## Reasoning Support

- Config: `[reasoning]` section with enable, thinking_effort (low/medium/high), exclude_from_output
- OpenAI: sets `reasoning_effort` in conversation, parses `reasoning_content` / `reasoning_details` from response
- Gemini: sets `thinkingConfig` with `thinkingBudget` (512/1024/2048), parses `thought: true` parts

## Streaming Captioning

`AsyncCaptionJob` (in `yadc/api/services/captioning/job.py`) is a pure state machine that implements `CaptioningCallbacks` and delegates to `AsyncCaptionJobRunner` for all API infrastructure. The runner (in `job_runner.py`) consumes `model.predict_stream()` token-by-token via `async for`, re-raising `CancelledError` to support mid-flight cancellation. The job's `request_stop()` cooperates with this — `CaptioningService.stop_job_async()` calls `job.wait(timeout=30)` after `request_stop()` so a new job can be started immediately. `_cleanup_async` is scheduled as a background task so `_arun` returns promptly.

Stream error handling: `predict`/`predict_stream` on both OpenAI and Gemini backends catch `httpx.RemoteProtocolError` / `httpx.ReadError`, log a warning, and raise a friendly `ValueError("Connection closed unexpectedly by the server. The API may have shut down or become unreachable.")` so the webui can surface a useful toast instead of an opaque httpx traceback.
