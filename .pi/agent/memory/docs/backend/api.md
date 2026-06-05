---
name: backend/api
description: Web UI backend — yadc/api/ Quart application with controllers/, modules/, and services/ sub-folders. The DI container, HTTP endpoints, watchers, and business logic all live here.
category: architecture
---

# Backend: Web UI API

The `yadc/api/` Quart web UI backend. Uses injector DI with auto-discovery (see `api-di-system`) and the service/repo pattern (see `repository-pattern`).

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
    app_frontend.py     # @controller — serves SvelteKit build
    api_datasets.py     # @controller — dataset/image endpoints (wired to DatasetService)
    api_captioning.py   # @controller — captioning start/stop/status + per-image caption (uses CaptioningService only — the SSE stream itself is in `api_events.py`)
    api_configs.py      # @controller — dataset config TOML CRUD (view/edit/delete)
    api_envs.py         # @controller — environment CRUD + model list proxy
    api_export.py       # @controller — export backends listing + run export
    api_templates.py    # @controller — template CRUD + Jinja2 variable extraction
    api_events.py       # @controller — SSE event stream with Last-Event-ID resumption support (replays from ring buffer on reconnect)

  modules/            # DI primitives + cross-cutting infrastructure
    __init__.py           # (package marker; no auto-discovery here — only `service.py` is the DI marker)
    service.py            # base Service class (marker for DI auto-discovery)
    cors_middleware.py    # CORSMiddleware — origin-based CORS, registered on ApiBlueprint
    logging_factory.py    # LoggingFactory — get_logger()
    event_dispatcher.py   # EventDispatcher — subscribe/dispatch + @event_handler decorator
    job_scheduler.py      # JobScheduler — daemon threads for periodic jobs
    sse_events.py         # SSEEvents — Condition-based SSE queue with ping, monotonic event IDs, ring buffer history for Last-Event-ID resumption
    env_watcher.py        # EnvWatcherService — watchdog-based watcher for config.toml, emits EnvironmentsChangedEvent (extends SinglePathWatcherService)
    template_watcher.py   # TemplateWatcherService — watchdog-based watcher for *.jinja files, emits TemplatesChangedEvent (extends SinglePathWatcherService)
    dataset_watcher.py    # DatasetWatcherService — watchdog-based filesystem watcher for dataset dirs, debounced DatasetChangedEvent emission
    db_migrations.py      # Step-based SQLite migration runner
    db_connection_factory.py # SQLite WAL, foreign keys, background init. `connection()` is a context manager that auto-enrolls in any active `transaction()`; standalone calls commit on success and close on exit. `transaction()` supports nested savepoints and is async/task-safe via `ContextVar`.

  services/           # Business logic + repositories (both auto-discovered as Service subclasses)
    __init__.py           # re-exports all services + repositories (CaptioningService, CaptionJobOptions, ConfigHistoryService, ConfigHistoryRepository, DatasetService, DatasetRepository, DatasetUploadService, DatasetUploadResult, ManagedDatasetsService, SettingsService, SettingsRepository, UploadProgressEvent)
    captioning.py         # CaptioningService — manages background captioning jobs (start/stop/status). `AsyncCaptionJob` implements `CaptioningCallbacks` and delegates the model-create / stream / save loop to `yadc.core.captioning.CaptioningRunner`; the job itself owns job state, SSE event emission (via EventDispatcher), job cancellation, and the final rescan. Config loading uses `yadc.core.captioning.load_dataset_config`.
    config_history.py     # ConfigHistoryService — high-level config revision operations; delegates SQL to `ConfigHistoryRepository`
    config_history_repository.py # ConfigHistoryRepository — owns `ConfigHistoryEntry` dataclass + all SQL for the `config_history` table (reads + writes + pruning)
    dataset_repository.py # DatasetRepository — owns `DatasetInfo` + `ImageInfo` dataclasses + all SQL for `datasets` / `dataset_images` tables (list, get, upsert, delete, scan diff)
    dataset_upload.py     # DatasetUploadService — orchestration layer for uploads (`create_dataset_from_upload`, `append_dataset_from_upload`, `commit_staged_upload`); thin wrappers around helpers in `dataset_upload_validation` (image/TOML validation, unique paths) and `dataset_upload_staging` (staging dir cleanup, conflict detection, grouped-build, orphan sidecar filtering). Injects `DatasetWatcherService` and accepts a `source` kwarg on all three methods; calls `expect_file_change(...)` before each file write/move and threads the source into the final `rescan_dataset(...)` so the originating tab's `DatasetChangedEvent` is suppressible.
    dataset_upload_staging.py    # Stateless staging helpers: `cleanup_staging_dirs`, `build_staged_groups`, `detect_conflicts`, `filter_orphan_sidecars`, `find_live_image_stems`. Constants: `STAGING_DIR_NAME = ".staging"`, `STAGING_MAX_AGE_SECONDS`.
    dataset_upload_validation.py # Stateless validation helpers: `validate_image_stream`, `validate_toml_stream`, `unique_path`.
    datasets.py           # DatasetService — TOML-based datasets, filesystem scanning, SQLite indexing, paginated image queries, caption read/write, import/create/delete/rescan, watches dirs via DatasetWatcherService. Injects `DatasetRepository` (no SQL in this file). Background refresh via `JobScheduler` (interval from `Configuration.dataset_refresh_interval_seconds`); `_refresh_lock` serializes background + manual rescans. `_apply_disk_scan` does a diff scan (no per-image upsert when nothing changed) and returns `bool` (whether the index changed). `rescan_dataset(source=...)` dispatches `DatasetChangedEvent(job_id=source)` when the scan changed rows.
    managed_datasets.py   # ManagedDatasetsService — operations specific to managed (upload-sourced) datasets: `list_folders` (images/ + folders/* with image counts + can_delete flag) and `delete_items` (file + sidecars, or folder, with `expect_file_change`/`expect_pattern_change` for self-suppression)
    managed_paths.py      # Layout constants and path builders for managed datasets: `MANAGED_IMAGES_PREFIX = "images"`, `MANAGED_FOLDERS_PREFIX = "folders"`, plus `managed_base_dir(config_path)`/`managed_images_dir(config_path)`/`managed_folders_dir(config_path)` builders and `compute_delete_path(image_path, config_path)` resolver. Single source of truth for the `<config_dir>/{images,folders}/...` layout — safe to import from anywhere (no service deps).
    settings.py           # SettingsService — KV store over SQLite settings table (JSON encode/decode + business policy); delegates SQL to `SettingsRepository`
    settings_repository.py # SettingsRepository — owns all SQL for the `settings` table (get, upsert, delete, list_all). Values are stored as raw JSON strings.
```

**Cross-references:**
- For DI auto-discovery: `api-di-system`
- For service/repo split: `repository-pattern`
- For dataset subsystem details: `dataset-system` and `dataset-watcher`
- For captioner backend: see `captioners.md`
