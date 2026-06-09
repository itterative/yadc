---
name: backend/captioners
description: API captioner implementations under yadc/captioners/ — APICaptioner, base, async_session, per-backend (openai, gemini, vllm, etc.), and shared utils.
category: architecture
---

# Backend: API Captioners (`yadc/captioners/`)

The actual captioner implementations. For the hierarchy, mixin pattern, streaming, and per-backend behavior, see `captioner-architecture`.

## See also (in `.pi/agent/memory/docs/`)

- `captioner-architecture` — APICaptioner hierarchy, mixin pattern, streaming, per-backend image limits
- `paths-and-storage` — HTTP response cache (file-based JSON keyed by SHA256)
- `debug-api-logging` — Debug logging (`YADC_DEBUG_CAPTION_RESPONSES`)

```
yadc/captioners/
  api/
    __init__.py       # (package marker)
    api_captioner.py  # APICaptioner — auto-detects API type, delegates to inner captioner
    base.py           # BaseAPICaptioner — async_session, cache, response_logger setup
    async_session.py  # Async HTTP Session with retries, caching, capture_response for debug logging. Configurable connect/read/write/pool timeouts (defaults: 30s connect, None read, 30s write/pool) threaded through from `Configuration.http_timeout_*`.
    openai.py         # OpenAI/OpenRouter/local backends (chat/completions). Both `predict` and `predict_stream` catch `httpx.RemoteProtocolError`/`ReadError` and surface a friendly "Connection closed unexpectedly" error.
    gemini.py         # Google Gemini backend (generateContent). Same stream error handling as OpenAI.
    koboldcpp.py      # KoboldCpp backend
    llamacpp.py       # llama.cpp backend
    ollama.py         # Ollama backend
    openrouter.py     # OpenRouter backend
    vllm.py           # vLLM backend
    types.py          # Pydantic models for all API response types
    utils/            # shared mixins and utilities (cache, error normalization, response logger, thinking, units)
```
