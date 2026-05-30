---
name: architecture-overview
description: High-level project structure, module organization, and key patterns. Start here for structural work or cross-cutting changes.
category: architecture
priority: 1
keep_updated: true
---

# yadc Architecture Overview

## What is yadc

Yet Another Dataset Captioner — a CLI tool for captioning image datasets using remote AI APIs (OpenAI, Gemini, OpenRouter, vLLM, llama.cpp, koboldcpp, Ollama).

## Package Structure

All source code lives under `yadc/` in the repo root.

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
  cli_webui.py        # web UI CLI (yadc webui serve)

  api/                # web UI backend (Quart + injector DI with auto-discovery)
    __init__.py
    application.py      # Application(Module) — DI container, auto-discovers services + controllers
    banner.txt           # ASCII art banner printed on startup (optional, --no-banner to disable)
    configuration.py    # @dataclass config (http, cors, sse, yadc paths, banner_enable)
    discovery.py        # discover_services() / discover_controllers() — package scanning
    watcher_base.py     # SinglePathWatcherService — shared base for filesystem watchers (debounce, observer lifecycle, create_event for subclasses)
    events.py           # Event base class + StartupEvent, ShutdownEvent, PingEvent, CaptioningStatusEvent, DatasetChangedEvent, ResumptionFailedEvent, EnvironmentsChangedEvent, TemplatesChangedEvent, ImageCaptionedEvent (with caption text)
    controllers/
      utils_json.py       # DataclassJSONEncoder + jsonify_dataclass + jsonify_error() (shared JSON utilities)
      models_errors.py    # APIErrorDetail + APIErrorResponse dataclasses, Pydantic ValidationError conversion
      __init__.py          # @controller decorator (auto-discovery marker + @inject)
      blueprints.py     # ApiBlueprint, AppBlueprint (@singleton injector classes)
      app_frontend.py   # @controller — serves SvelteKit build
      api_datasets.py   # @controller — dataset/image endpoints (wired to DatasetService)
      api_captioning.py # @controller — captioning start/stop/status, SSE stream via SSEEvents + CaptioningService
      api_configs.py   # @controller — dataset config TOML CRUD (view/edit/delete)
      api_envs.py       # @controller — environment CRUD + model list proxy
      api_export.py     # @controller — export backends listing + run export
      api_templates.py  # @controller — template CRUD + Jinja2 variable extraction
      api_events.py     # @controller — SSE event stream with Last-Event-ID resumption support (replays from ring buffer on reconnect)
    modules/
      service.py            # base Service class (marker for DI auto-discovery)
      cors_middleware.py    # CORSMiddleware — origin-based CORS, registered on ApiBlueprint
      logging_factory.py    # LoggingFactory — get_logger()
      event_dispatcher.py   # EventDispatcher — subscribe/dispatch + @event_handler decorator
      job_scheduler.py       # JobScheduler — daemon threads for periodic jobs
      sse_events.py         # SSEEvents — Condition-based SSE queue with ping, monotonic event IDs, ring buffer history for Last-Event-ID resumption
      env_watcher.py       # EnvWatcherService — watchdog-based watcher for config.toml, emits EnvironmentsChangedEvent (extends SinglePathWatcherService)
      template_watcher.py  # TemplateWatcherService — watchdog-based watcher for *.jinja files, emits TemplatesChangedEvent (extends SinglePathWatcherService)
      dataset_watcher.py    # DatasetWatcherService — watchdog-based filesystem watcher for dataset dirs, debounced DatasetChangedEvent emission
      db_migrations.py      # Step-based SQLite migration runner
      db_connection_factory.py # SQLite WAL, foreign keys, background init
    services/
      __init__.py           # re-exports CaptioningService, DatasetService, DatasetUploadService, DatasetUploadResult, SettingsService
      captioning.py        # CaptioningService — background captioning jobs (start/stop/status), env/config/template resolution, CaptioningStatusEvent emission via EventDispatcher
      dataset_upload.py     # DatasetUploadService — file upload validation, writing, and dataset registration (extracted from DatasetService)
      datasets.py           # DatasetService — TOML-based datasets, filesystem scanning, SQLite indexing, paginated image queries, caption read/write, import/create/delete/rescan, watches dirs via DatasetWatcherService
      settings.py           # SettingsService — KV store over SQLite settings table (JSON values)

  webui/             # SvelteKit frontend — see `frontend-architecture` memory for full details
    …

  cmd/                # pure logic (no click imports)
    app.py            # paths (CONFIG_PATH, STATE_PATH, CACHE_PATH via platformdirs), load_config()
    config.py         # AppConfig Pydantic model hierarchy (v0→v1 TOML migration), save_config()
    status.py         # exit codes: STATUS_OK=0, STATUS_ERROR=1, STATUS_USER_ERROR=2
    cache/            # cache dir helpers, clean_cache()
    configs/          # user config CRUD, deep merge
    envs/             # env loading/saving, RSA encryption via keyring or password
      encryption.py   # active KeyStorage management, RSA encrypt/decrypt, key-mode switch
      envs.py         # env CRUD using AppConfig
      setting.py      # Setting base class, EncryptionMethod enum
      keystorage.py   # KeyStorage ABC (load/save private key, generate key pair)
      keystorage_keyring.py   # KeyringKeyStorage — system keyring private key
      keystorage_password.py  # PasswordKeyStorage — config-TOML private key, PBKDF2 + AES-256-GCM
      user_config.py  # UserConfig / UserConfigApi models (legacy)
    templates/        # user template CRUD in STATE_PATH/templates/

  utils/              # shared utility functions
    dict_utils.py      # deep_merge() for TOML config patching

  core/               # business logic
    captioner.py      # Abstract Captioner base class (Jinja2 prompts, image encoding)
    config.py         # Config models (v1/v2), parse_config()
    dataset.py        # DatasetImage model (path, caption, drafts, history, TOML persistence)
    dataset_resolver.py # resolve_dataset() — scans paths, merges images, applies extras
    env.py            # env var flags (DEBUG_CAPTION_RESPONSES, YADC_PASSWORD, etc.)
    exporters/        # export backends (currently sd-scripts: json/jsonl/txt)
      utils.py          # read_caption_source() — shared caption/draft reading for export
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
- **DatasetImage persistence**: `.txt` for caption, `.toml` for metadata extras, `.history~` for versioned history (TOML entries separated by `----------` markers), `.<name>.draft~` for named drafts. History is saved on every caption update (both captioning jobs and manual webui edits) and extras updates. History entries can be browsed and restored via `GET /images/<id>/history` and `PUT /images/<id>/history/<index>/restore`.
- **Platformdirs paths**: Config → `~/.config/yadc/`, State → `~/.local/state/yadc/`, Cache → `~/.cache/yadc/`
- **Web UI DI with auto-discovery**: See `api-di-system` memory for full details. Short version: `Service` subclasses in `modules/` **and `services/`** and `@controller` functions in `controllers/` are auto-discovered — no hardcoded lists. Services are plain classes (no decorators), controllers use `@controller` from `controllers/__init__.py`.
- **Web UI dataset model**: A "dataset" IS a TOML config file at `STATE_PATH/<name>/config.toml`. Three creation flows:
  - `import_dataset(name, toml_path)` — copies existing TOML to state dir (resolving relative paths).
  - `create_dataset(name, image_paths)` — generates a new TOML pointing to external directories.
  - `create_dataset_from_upload(name, files)` — creates a self-contained dataset from uploaded files (images + sidecars). Stores files in `STATE_PATH/<name>/images/` (root files flat) and `STATE_PATH/<name>/folders/` (folder uploads, one subdir per top-level folder). Returns `DatasetUploadResult` (dataset + warnings list).
  - `name` is the unique key — no `path` column.
- **Upload endpoint**: `POST /api/datasets/upload` accepts `multipart/form-data` with `name` + `files`. Two-stage image validation (`Image.verify()` fast check, `Image.load()` fallback), TOML syntax validation, orphan sidecar detection, atomic cleanup on failure. Configurable `max_upload_size_bytes` limit.
- **Hash routing**: SvelteKit uses `router: { type: "hash" }` — all internal links use `#/` prefix. Quart only serves `GET /` + static assets.
- **WebUI frontend**: See `frontend-architecture` memory for full directory structure, stores, components, and frontend-specific patterns.
- **Config validation is strict by default, relaxable**: `parse_config(raw)` enforces CLI-level checks (api url/model_name must exist, prompt must be specified). `parse_config(raw, strict=False)` skips those checks (used by webui, which provides these at caption time). Uses Pydantic validation context to thread the flag — no fields on the model. The `ConfigV1.to_v2()` uses `model_construct()` to avoid re-running validators on already-validated data.
- **npm security**: `min-release-age=14` in `.npmrc` blocks installing packages published <14 days ago.
- **WebUI CLI**: `yadc webui serve` — defaults to host `127.0.0.1`, port `7860`. Run in tmux for background dev: `tmux new-session -d -s yadc-webui "uv run yadc webui serve --host 127.0.0.1"`
