---
name: backend/captioners
description: The single APICaptioner (captioners/api/) — composes a BaseLLMClient from yadc/llm/ and holds all captioning glue (image encode, Jinja, conversation overrides, thinking strip).
category: architecture
---

# Backend: Captioners (`yadc/captioners/`)

The captioning layer. A single `APICaptioner` composes a
`BaseLLMClient` (from `yadc/llm/`). The client owns all wire-format /
streaming / reasoning logic; the captioner owns captioning-specific glue.

## See also (in `.pi/agent/memory/docs/`)

- `captioner-architecture` — hierarchy, ThinkingMixin, conversation building, PredictionContext (now captioner-side)
- `backend/llm` — the `yadc/llm/` client layer the captioner composes
- `paths-and-storage` — HTTP response cache
- `debug-api-logging` — Debug logging (`YADC_DEBUG_CAPTION_RESPONSES`)

```
yadc/captioners/
  api/
    __init__.py       # re-exports APICaptioner, APITypes
    api_captioner.py  # APICaptioner(Captioner, ThinkingMixin) — composes a BaseLLMClient;
                      #   image limits + api_type resolved from the client class;
                      #   load_model/unload/offload/log_usage/list_models proxy to the client.
    constants.py      # DEFAULT_LIST_MODELS_TIMEOUT_SECONDS (env-orchestrator timeout; DEFAULT_MODELS_CACHE_TTL_SECONDS moved to yadc.llm.constants)
    utils/
      thinking.py     # ThinkingMixin — strips inline <think> tags from output
```

All shared HTTP infra (`AsyncSession`, `HTTPResponseCache`,
`ResponseLogger`, `normalize_error`, `size_units`, response models,
`DEFAULT_MODELS_CACHE_TTL_SECONDS`) lives in `yadc/llm/`. The
captioner layer depends on `yadc/llm/`, never the reverse.
