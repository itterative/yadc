---
name: captioner-architecture
description: How the captioner hierarchy works — APICaptioner auto-detection, inner captioner delegation, mixin pattern, and per-backend details.
---

# Captioner Architecture

## Hierarchy

```
Captioner (abc)                        # core/captioner.py
  ├─ load_model(), unload_model(), offload_model()
  ├─ predict(), predict_stream()
  ├─ prompts_from_image()              # Jinja2 template rendering
  └─ _encode_image()                   # base64 with auto-resize/quality degradation

BaseAPICaptioner (abc)                 # captioners/api/base.py
  └─ Sets up Session, HTTPResponseCache, ResponseLogger
  └─ _before_predict() resets PredictionContext

APICaptioner                          # captioners/api/api_captioner.py
  └─ Auto-detects API type → delegates to inner_captioner

OpenAICaptioner / GeminiCaptioner / etc.  # per-backend
  └─ BaseAPICaptioner + ErrorNormalizationMixin + ThinkingMixin
```

## API Type Auto-Detection (`APICaptioner._infer_api_type()`)

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

## Mixin Pattern

- **ErrorNormalizationMixin**: `_normalize_error(error)` — handles HTTPError, GenerationError, Pydantic response objects. Parses OpenAI/Gemini error JSON, OpenRouter moderation errors. Falls back to HTTP status code messages.
- **ThinkingMixin**: `_handle_thinking(content)` / `_handle_thinking_streaming(stream)` — strips `<thinking_start>...<thinking_end>` tokens from output, logs thinking content with `> ` prefix.

## Conversation Building

Each backend builds an API-specific conversation dict from `prompts_from_image()` + `_encode_image()`:
- **OpenAI**: `chat/completions` with messages array (system, user with image_url + text, optional assistant prefill, extra_messages as ReplyRounds)
- **Gemini**: `models/{model}:generateContent` with system_instruction, contents, generationConfig, optional thinkingConfig and safetySettings

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
