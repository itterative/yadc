---
name: backend/api
description: Web UI backend — yadc/api/ Quart application with controllers/, modules/, and services/ sub-folders. The DI container, HTTP endpoints, watchers, and business logic all live here.
category: architecture
---

# Backend: Web UI API

The `yadc/api/` Quart web UI backend. Uses injector DI with auto-discovery (see `api-di-system`) and the service/repo pattern (see `repository-pattern`).

## See also (in `.pi/agent/memory/docs/`)

- `api-di-system` — DI auto-discovery (services, controllers, injector binding)
- `repository-pattern` — Service/repo split for SQL
- `dataset-system` — Dataset subsystem (model, creation, upload, persistence)
- `dataset-watcher` — Inotify watcher + expected-change tracking
- `captioners` — Per-backend API captioners (in `backend/`)

```
yadc/api/
  __init__.py
  application.py      # Application(Module) — DI container, auto-discovers services + controllers
  banner.txt          # ASCII art banner printed on startup (optional, --no-banner to disable)
  configuration.py    # @dataclass config (http, cors, sse, watcher, http timeouts, dataset refresh, upload size, yadc paths, banner_enable)
  discovery.py        # discover_services() / discover_controllers() — package scanning
  watcher_base.py     # SinglePathWatcherService — shared base for filesystem watchers (debounce, observer lifecycle, create_event for subclasses)
  events.py           # Event base class + StartupEvent, ShutdownEvent, PingEvent, CaptioningStatusEvent, DatasetChangedEvent, ResumptionFailedEvent, EnvironmentsChangedEvent, TemplatesChangedEvent, ImageCaptionedEvent (with caption text), ImageCaptionStartedEvent, ImageCaptionErrorEvent

  controllers/        # HTTP endpoints — @controller + @inject, auto-discovered
    __init__.py         # @controller decorator (auto-discovery marker + @inject)
    utils_json.py       # DataclassJSONEncoder + jsonify_dataclass + jsonify_error() (shared JSON utilities)
    models_errors.py    # APIErrorDetail + APIErrorResponse dataclasses, Pydantic ValidationError conversion
    blueprints.py       # ApiBlueprint, AppBlueprint (@singleton injector classes)
    _password.py        # Shared cookie helpers (private): `resolve_request_password`, `set_password_cookie`, `clear_password_cookie`. All password-requiring controllers read the `yadc_password` session cookie via `resolve_request_password` (with `YADC_PASSWORD` env-var fallback).
    app_frontend.py     # @controller — serves SvelteKit build
    api_auth.py         # @controller — `POST/DELETE /api/auth/password` — validate password and set/clear the `yadc_password` session cookie
    api_datasets.py     # @controller — dataset/image endpoints (wired to DatasetService)
    api_captioning.py   # @controller — captioning start/stop/status + per-image caption + refine (uses CaptioningService only — the SSE stream itself is in `api_events.py`)
    api_configs.py      # @controller — dataset config TOML CRUD (view/edit/delete)
    api_envs.py         # @controller — environment CRUD + model list proxy
    api_export.py       # @controller — export backends listing + run export (filesystem or streaming zip)
    api_templates.py    # @controller — template CRUD + Jinja2 variable extraction
    api_prompts.py      # @controller — POST /api/prompts/generate — streaming NDJSON response that delegates to PromptGenerationService (no SSE event bus; uses fetch + ReadableStream on the frontend)
    api_prompts_history.py # @controller — `/api/prompts/history/...` — save / list / get / delete prompt-history entries. Pydantic body for the save, JSON-friendly dicts for the responses, opaque pagination cursor for the list. Reuses `ExamplePair` from the prompt_generation service.
    api_events.py       # @controller — SSE event stream with Last-Event-ID resumption support (replays from ring buffer on reconnect)

  modules/            # DI primitives + cross-cutting infrastructure
    __init__.py           # (package marker; no auto-discovery here — only `service.py` is the DI marker)
    service.py            # base Service class (marker for DI auto-discovery)
    cors_middleware.py    # CORSMiddleware — origin-based CORS, registered on ApiBlueprint
    logging_factory.py    # LoggingFactory — get_logger()
    uvicorn_logging_config.py  # UvicornLoggingConfig — mutates uvicorn.config.LOGGING_CONFIG during SetupAppEvent to apply the yadc format (from Configuration.logging_log_format) and the level for the main `uvicorn` logger (other uvicorn loggers get their level from Config.log_level, which Application.run() sources from LoggingFactory.log_level)
    event_dispatcher.py   # EventDispatcher — subscribe/dispatch + @event_handler decorator
    job_scheduler.py      # JobScheduler — daemon threads for periodic jobs
    sse_events.py         # SSEEvents — Condition-based SSE queue with ping, monotonic event IDs, ring buffer history for Last-Event-ID resumption
    env_watcher.py        # EnvWatcherService — watchdog-based watcher for config.toml, emits EnvironmentsChangedEvent (extends SinglePathWatcherService)
    template_watcher.py   # TemplateWatcherService — watchdog-based watcher for *.jinja files, emits TemplatesChangedEvent (extends SinglePathWatcherService)
    dataset_watcher.py    # DatasetWatcherService — watchdog-based filesystem watcher for dataset dirs, debounced DatasetChangedEvent emission
    db_migrations.py      # File-based SQLite migration runner with rollback support, transactional upgrades, and auto-downgrade on app rollback. Migrations are numbered SQL file pairs `<NNNN>_<name>_{up,down}.sql` in `yadc.api.migrations`. Optionally pairs with sibling `<NNNN>_<name>_{up,down}.py` Python hooks for **data migrations** — the hook module's `migrate(conn)` function runs inside the same `BEGIN IMMEDIATE` transaction as the SQL step (atomicity guaranteed; a Python exception rolls back the SQL change too). Most migrations stay pure SQL — the hook is purely additive and `None` when no sibling `.py` exists. Hook modules are discovered lazily on first use via `importlib.import_module`. At upgrade time the **whole hook module source** (not just `def migrate`) is captured into `migration_downgrades_py` so rollbacks work when the package source is gone (function-only capture loses imports + `from __future__ import annotations` and crashes on re-`exec`). On rollback the stored source is re-`exec`d via `sandbox.py`. No migration currently ships a Python hook — the machinery is latent and exercised only by the `_hook_capture_fixture` module + the synthetic drift tests; it's kept for the next real data migration.
    sandbox.py            # Restricted-exec sandbox for the stored Python migration hook source (`exec_hook(source, module_name)`). Reduced `__builtins__` (drops `open`/`compile`/`eval`/`exec`) + whitelist-only `__import__` (allowed stdlib: `base64`, `json`, `re`, `sqlite3`, etc.; keep additive-only). Defense-in-depth against a corrupted stored row, NOT a security boundary — see the `db_migrations` threat model (local DB, attacker-with-DB-write also has package access).
    db_connection_factory.py # SQLite WAL, foreign keys, background init. `connection()` is a context manager that auto-enrolls in any active `transaction()`; standalone calls commit on success and close on exit. `transaction()` supports nested savepoints and is async/task-safe via `ContextVar`.

  services/           # Business logic + repositories (both auto-discovered as Service subclasses)
    __init__.py           # re-exports all services and repositories (CaptioningService, CaptionJobOptions, ConfigHistoryService, ConfigHistoryRepository, DatasetService, DatasetRepository, DatasetUploadService, DatasetUploadResult, ExamplePair, ManagedDatasetsService, PromptGenerationFocus, PromptGenerationRequest, PromptGenerationService, SettingsService, SettingsRepository, UploadProgressEvent)
    captioning/           # CaptioningService + job management (package, split from single file)
      __init__.py           # re-exports CaptioningService, AsyncCaptionJob, AsyncCaptionJobRunner, RefineOptions, JobStatus, JobInfo
      models.py             # RefineOptions, JobStatus, JobInfo (data classes / type aliases)
      service.py            # CaptioningService — manages background captioning jobs (start/stop/status). Dispatches `CaptioningStatusEvent` for job-level transitions, runs final `DatasetService.rescan_dataset()` after completion.
      job_runner.py          # AsyncCaptionJobRunner — API glue between the service and the captioning core. Holds DI deps, does preflight (config loading, image resolution, DB ordering), SSE event emission, watcher registration. Created by CaptioningService per job.
      job.py                # AsyncCaptionJob — pure state machine + `CaptioningCallbacks` implementation. No DI imports; delegates all infrastructure to `AsyncCaptionJobRunner`. Owns lifecycle (start/stop/wait), state tracking, snapshot.
    config_history.py     # ConfigHistoryService — high-level config revision operations; delegates SQL to `ConfigHistoryRepository`
    config_history_repository.py # ConfigHistoryRepository — owns `ConfigHistoryEntry` dataclass + all SQL for the `config_history` table (reads + writes + pruning)
    dataset_repository.py # DatasetRepository — owns `DatasetInfo` + `ImageInfo` dataclasses + all SQL for `datasets` / `dataset_images` tables (list, get, upsert, delete, scan diff)
    dataset_upload.py     # DatasetUploadService — orchestration layer for uploads (`create_dataset_from_upload`, `append_dataset_from_upload`, `commit_staged_upload`); thin wrappers around helpers in `dataset_upload_validation` (image/TOML validation, unique paths) and `dataset_upload_staging` (staging dir cleanup, conflict detection, grouped-build, orphan sidecar filtering). Injects `DatasetWatcherService` and accepts a `source` kwarg on all three methods; calls `expect_file_change(...)` before each file write/move and threads the source into the final `rescan_dataset(...)` so the originating tab's `DatasetChangedEvent` is suppressible.
    dataset_upload_staging.py    # Stateless staging helpers: `cleanup_staging_dirs`, `build_staged_groups`, `detect_conflicts`, `filter_orphan_sidecars`, `find_live_image_stems`. Constants: `STAGING_DIR_NAME = ".staging"`, `STAGING_MAX_AGE_SECONDS`.
    dataset_upload_validation.py # Stateless validation helpers: `validate_image_stream`, `validate_toml_stream`, `unique_path`.
    datasets.py           # DatasetService — TOML-based datasets, filesystem scanning, SQLite indexing, paginated image queries, caption read/write, import/create/delete/rescan, watches dirs via DatasetWatcherService. Injects `DatasetRepository` (no SQL in this file). Background refresh via `JobScheduler` (interval from `Configuration.dataset_refresh_interval_seconds`); `_refresh_lock` serializes background + manual rescans. `_apply_disk_scan` does a diff scan (no per-image upsert when nothing changed) and returns `bool` (whether the index changed). `rescan_dataset(source=...)` dispatches `DatasetChangedEvent(job_id=source)` when the scan changed rows.
    managed_datasets.py   # ManagedDatasetsService — operations specific to managed (upload-sourced) datasets: `list_folders` (images/ + folders/* with image counts + can_delete flag) and `delete_items` (file + sidecars, or folder, with `expect_file_change`/`expect_pattern_change` for self-suppression)
    managed_paths.py      # Layout constants and path builders for managed datasets: `MANAGED_IMAGES_PREFIX = "images"`, `MANAGED_FOLDERS_PREFIX = "folders"`, plus `managed_base_dir(config_path)`/`managed_images_dir(config_path)`/`managed_folders_dir(config_path)` builders and `compute_delete_path(image_path, config_path)` resolver. Single source of truth for the `<config_dir>/{images,folders}/...` layout — safe to import from anywhere (no service deps).
    prompt_generation/    # PromptGenerationService — one-shot Jinja2 prompt template generator. Builds the LLM client directly via `yadc.llm.create_client` (no captioner — there's no image). Yields `StreamChunk`s from the client's `predict_next_message_stream(messages, reasoning=None)`. Used by the `api_prompts` controller for the prompt-generator feature.
      __init__.py         # Package entry point: re-exports the public surface (ExamplePair, PromptGenerationFocus, PromptGenerationRequest, PromptGenerationService) and re-exports cmd_envs + create_client at the package level for test-patch compatibility. Pure re-export — does NOT load the system prompt (that would be a circular import with `service.py`).
      service.py          # The actual service + `_build_messages` + `_EXAMPLE_ACK` + the private `_GENERATE_SYSTEM_PROMPT` constant (loaded once at import time from `prompts/generate.txt` via `importlib.resources`). The constant lives in `service.py` (not `__init__.py`) because it's the only consumer — putting it in the package would force `service.py` to import the package back, creating a circular import. `PromptGenerationRequest` carries the generation limits — `max_tokens` (default 16384; the client default of 4096 truncates longer templates) and `image_quality` (auto/low/high). `generate` forwards `max_tokens` to `predict_next_message_stream` and threads `image_quality` both into `_build_messages` (→ OpenAI `image_url.detail` on the example image parts) AND as a client opt (→ Gemini `mediaResolution`), mirroring how the captioner covers both backends. `_build_messages(request, *, system_prompt, refine_system_prompt, image_quality)` prepends `Message(role="system", content=system_prompt)`; `generate` passes `_GENERATE_SYSTEM_PROMPT` (or `_REFINE_SYSTEM_PROMPT` in refine mode).
      prompts/            # Data directory for system prompts. No `__init__.py`. Loaded via `importlib.resources` so the files travel with the package (works for uv editable, wheel, and zip-imports — `__file__`-relative paths break in the latter two).
        generate.txt      # The `_GENERATE_SYSTEM_PROMPT` text — the system prompt sent to the model during generation (prepended to the message list in `service.py` via `_build_messages`).
        refine.txt        # The `_REFINE_SYSTEM_PROMPT` text — the system prompt for the refine-mode meta-conversation (sent when `request.template_content is not None`).
    prompt_history.py           # PromptHistoryService — save / list / get / delete prompt-history entries. Owns the `PROMPT_HISTORY_MAX_ENTRIES = 20` cap, the prune-on-save policy (insert entry + insert examples + prune in one transaction), the **base64 ↔ BLOB translation at the wire boundary** (the wire format is `ExamplePair.image_data_url`; the repo stores raw image bytes in the sibling `prompt_example_images` BLOB table — no base64 inflation in the DB), and the server-derived list-view fields (`intent_preview`, `example_count`, `had_template`). `PromptHistorySaveRequest` is the service-layer Pydantic model (re-validated from the controller body for self-contained service tests). `PromptHistoryListItem` + `PromptHistoryPage` are the list-response dataclasses; opaque `next` cursor matches the `config_history` pattern (`2**63 - 1` for the unbounded case). `get_entry` returns a JSON-friendly dict (with `examples` re-encoded as `data:<mime>;base64,...` URLs from the BLOB rows) so the controller can `jsonify` it directly. Malformed `image_data_url` values are silently skipped on save (one bad example shouldn't block the whole save).
    prompt_history_repository.py # PromptHistoryRepository — owns `PromptHistoryEntry` (entry-level row) + `PromptExample` (per-image row, child of an entry) dataclasses and all SQL for both tables. `list_entries_with_counts` is a single `LEFT JOIN ... GROUP BY` so `example_count` comes back attached (no per-row JSON parse). `get_entry_with_examples` returns `(entry, examples)` from one connection (shared transactional snapshot). `insert_entry` + `insert_examples` + `prune_old` are called by the service inside one `with db.transaction():` block for save atomicity. `delete_entry` cascades to image rows via the FK `ON DELETE CASCADE` — no service-level cleanup. Inline row→dataclass construction per the repo convention. `prune_old(keep_count)` is a single `DELETE WHERE id NOT IN (SELECT id ORDER BY id DESC LIMIT ?)` — keeps the `keep_count` newest rows, deletes everything else.
    settings.py           # SettingsService — KV store over SQLite settings table (JSON encode/decode + business policy); delegates SQL to `SettingsRepository`
    settings_repository.py # SettingsRepository — owns all SQL for the `settings` table (get, upsert, delete, list_all). Values are stored as raw JSON strings.
```
