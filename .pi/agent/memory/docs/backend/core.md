---
name: backend/core
description: Business logic under yadc/core/ — Captioner base, Config models, Dataset model, dataset resolution, env flags, exporters, logging, prediction context.
category: architecture
---

# Backend: Business Logic (`yadc/core/`)

The model layer and core business logic. The API backend (`yadc/api/`) wraps these with HTTP endpoints and DI.

## See also (in `.pi/agent/memory/docs/`)

- `yadc-config-v2` — Config models and v1/v2 format
- `captioner-architecture` — Captioner base + per-backend details
- `captioning-runner` — Shared captioning runner (CLI + API)
- `captioning-workflow` — End-to-end captioning flow
- `paths-and-storage` — DatasetImage file conventions
- `dataset-system` — DatasetImage endpoints
- `export-system` — Exporters

```
yadc/core/
  captioner.py        # Abstract Captioner base class (Jinja2 prompts, image encoding)
  captioning/         # Shared captioning core — used by both the CLI and the API
    options.py        #   CaptionJobOptions (Pydantic model)
    loader.py         #   apply_config_overrides, resolve_template, load_dataset_config
    runner.py         #   CaptioningRunner (async context manager) + CaptioningCallbacks (Protocol)
  config.py           # Config models (v1/v2), parse_config()
  dataset.py          # DatasetImage model (path, caption, drafts, history, TOML persistence)
  dataset_resolver.py # resolve_dataset() — scans paths, merges images, applies extras
  env.py              # env var flags (DEBUG_CAPTION_RESPONSES, YADC_PASSWORD, etc.)
  exporters/          # export backends (currently sd-scripts: json/jsonl/txt)
    sd_scripts.py     # sd-scripts export backend (json/jsonl/txt)
    utils.py          # read_caption_source() — shared caption/draft reading for export
  logging.py          # custom logger with TRACE level, global level/handler management
  prediction.py       # PredictionContext — mutable container for reasoning data
  user_config.py      # UserConfig / UserConfigApi models
  utils.py            # Timer context manager
```
