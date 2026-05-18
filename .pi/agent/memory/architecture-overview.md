---
name: architecture-overview
description: High-level project architecture and module organization for yadc.
---

# yadc Architecture Overview

## What is yadc

Yet Another Dataset Captioner — a CLI tool for captioning image datasets using remote AI APIs (OpenAI, Gemini, OpenRouter, vLLM, llama.cpp, koboldcpp, Ollama).

## Package Structure

```
yadc/
  __init__.py         # version
  __main__.py         # entry point → yadc.cli:cli
  cli.py              # click group registration, version command

  cli_caption.py      # main captioning command (largest CLI module)
  cli_cache.py        # cache management CLI
  cli_common.py       # shared click options (--log-level, --env)
  cli_configs.py      # user configs management CLI
  cli_draft.py        # draft save/list/show/remove CLI
  cli_envs.py         # environment settings CLI
  cli_export.py       # export captions to training formats
  cli_logging.py      # ClickHandler — routes logs through click.secho
  cli_templates.py    # template management CLI

  cmd/                # pure logic (no click imports)
    app.py            # paths (CONFIG_PATH, STATE_PATH, CACHE_PATH via platformdirs), load_config()
    status.py         # exit codes: STATUS_OK=0, STATUS_ERROR=1, STATUS_USER_ERROR=2
    cache/            # cache dir helpers, clean_cache()
    configs/          # user config CRUD, deep merge
    envs/             # env loading/saving, RSA encryption via keyring
    templates/        # user template CRUD in STATE_PATH/templates/

  core/               # business logic
    captioner.py      # Abstract Captioner base class (Jinja2 prompts, image encoding)
    config.py         # Config models (v1/v2), parse_config()
    dataset.py        # DatasetImage model (path, caption, drafts, history, TOML persistence)
    dataset_resolver.py # resolve_dataset() — scans paths, merges images, applies extras
    env.py            # env var flags (DEBUG_CAPTION_RESPONSES, etc.)
    exporters/        # export backends (currently sd-scripts: json/jsonl/txt)
    logging.py        # custom logger with TRACE level, global level/handler management
    prediction.py     # PredictionContext — mutable container for reasoning data
    user_config.py    # UserConfig / UserConfigApi models
    utils.py          # Timer context manager

  captioners/
    api/
      api_captioner.py  # APICaptioner — auto-detects API type, delegates to inner captioner
      base.py           # BaseAPICaptioner — session, cache, response_logger setup
      session.py        # HTTP Session with retries, caching, capture_response for debug logging
      openai.py         # OpenAI/OpenRouter/local backends (chat/completions)
      gemini.py         # Google Gemini backend (generateContent)
      koboldcpp.py      # KoboldCpp backend
      llamacpp.py       # llama.cpp backend
      ollama.py         # Ollama backend
      openrouter.py     # OpenRouter backend
      vllm.py           # vLLM backend
      types.py          # Pydantic models for all API response types
      utils/
        cache.py            # HTTPResponseCache — file-based SHA256 keyed JSON cache
        error_normalization.py # ErrorNormalizationMixin — normalizes API errors to strings
        response_logger.py  # ResponseLogger — JSONL debug logging
        thinking.py         # ThinkingMixin — strips thinking tokens from output
        units.py            # size_units() formatter

  templates/
    __init__.py        # re-exports default_template, load_builtin_template
    templates.py       # loads from yadc/templates/jinja/ via importlib.resources
    jinja/
      default.jinja    # built-in prompt template (system_prompt, user_prompt, user_prompt_multiple_rounds)
```

## Key Patterns

- **API type auto-detection**: `APICaptioner` infers the backend from URL domain and `/models` response, then delegates to the appropriate inner captioner
- **Mixin composition**: OpenAI/Gemini captioners use `ErrorNormalizationMixin` + `ThinkingMixin`
- **Jinja2 template system**: Templates define `{% set system_prompt %}`, `{% set user_prompt %}`, `{% set user_prompt_multiple_rounds %}` blocks. User templates override defaults.
- **DatasetImage persistence**: `.txt` for caption, `.toml` for metadata extras, `.history~` for versioned history, `.<name>.draft~` for named drafts
- **Platformdirs paths**: Config → `~/.config/yadc/`, State → `~/.local/state/yadc/`, Cache → `~/.cache/yadc/`
