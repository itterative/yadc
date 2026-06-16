"""``Configuration`` dataclass — every web UI server knob in one place.

Grouped by subsystem:

- **HTTP server** — ``http_host`` / ``http_port``, ``graceful_shutdown_timeout``,
  ``max_upload_size_bytes`` (also forwarded to Quart's
  ``MAX_CONTENT_LENGTH`` so request bodies > 16 MB can be parsed).
- **SvelteKit frontend** — ``app_frontend_build_path`` (where
  ``app_frontend`` serves files from) and ``app_frontend_cache_control``
  for ``/_app/...`` assets.
- **CORS** (dev mode) — origin, methods, headers, credentials, any-origin.
- **SSE** — ``sse_listeners_warning`` / ``sse_listeners_max`` (cap on
  concurrent clients), ``sse_listener_max_events`` (per-client queue
  size), ``sse_event_history_size`` (ring buffer for ``Last-Event-ID``
  resumption).
- **File watcher** — ``watcher_debounce_seconds`` and the
  expected-file ring buffer (``watcher_expected_file_max`` /
  ``watcher_expected_file_ttl``) used to suppress self-induced events.
- **Dataset index** — ``dataset_refresh_interval_seconds`` fallback
  scan for changes the inotify watcher can't see.
- **Captioning job cleanup** — periodic GC of finished jobs
  (``captioning_cleanup_interval_seconds`` + grace period).
- **Logging** — ``logging_default_level``, ``logging_log_format`` (applied
  to the ``default`` uvicorn formatter), ``access_log_file`` (per-request
  access log; ``RotatingFileHandler`` for the default cache path,
  ``WatchedFileHandler`` when the user supplies the path), and the
  rotation knobs ``access_log_max_bytes`` / ``access_log_backup_count``.
- **HTTP client timeouts** — connect/read/write/pool, threaded into
  ``AsyncSession``.
- **Caches** — ``api_models_cache_ttl`` for ``/api/envs/.../models``,
  ``refine_result_buffer_size`` for the refine endpoint.
- **Model-list timeout** — ``list_models_timeout`` caps the entire
  ``/api/envs/<name>/models`` operation so a dead/slow env doesn't
  leave the model picker spinning forever. Surfaces as HTTP 504.
- **Yadc paths** — ``config_path``, ``state_path``, ``cache_path``,
  ``db_path`` (SQLite for the web UI's datasets/configs/settings tables).
"""

from dataclasses import dataclass, field

from yadc.captioners.api.constants import (
    DEFAULT_LIST_MODELS_TIMEOUT_SECONDS,
    DEFAULT_MODELS_CACHE_TTL_SECONDS,
)
from yadc.cmd.app import CACHE_PATH, CONFIG_PATH, STATE_PATH


@dataclass
class Configuration:
    # HTTP
    http_host: str = "127.0.0.1"
    http_port: int = 7860

    # Frontend
    app_frontend_build_path: str = str(__import__("pathlib").Path(__file__).parent.parent / "webui" / "build")
    app_frontend_cache_control: str = "public, max-age=31536000, immutable"

    # CORS (dev mode)
    api_cors_enable: bool = True
    api_cors_allow_credentials: bool = True
    api_cors_allow_methods: str = "GET, POST, OPTIONS, PUT, PATCH, DELETE"
    api_cors_allow_headers: list[str] = field(default_factory=lambda: ["Content-Type"])
    api_cors_allow_any_origin: bool = True

    # Logging
    logging_default_level: int = 20  # logging.INFO
    logging_log_format: str = "%(asctime)s - %(levelname)s - %(message)s"
    # Per-request access log. UvicornLoggingConfig picks RotatingFileHandler
    # (default cache path) or WatchedFileHandler (user-supplied path) based
    # on access_log_user_specified; access lines always emit at INFO.
    access_log_file: str = field(default_factory=lambda: str(CACHE_PATH / "webui-access.log"))
    access_log_user_specified: bool = False
    access_log_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    access_log_backup_count: int = 5

    # SSE
    sse_listeners_warning: int = 25
    sse_listeners_max: int = 100
    sse_listener_max_events: int = 100
    sse_event_history_size: int = 128

    # Server
    graceful_shutdown_timeout: int = 5
    graceful_shutdown_threads_timeout: float = 1

    # Display
    banner_enable: bool = True

    # File watcher
    watcher_debounce_seconds: float = 1.0
    watcher_expected_file_max: int = 256
    watcher_expected_file_ttl: float = 1.0

    # Dataset index
    # How often the background thread re-scans datasets that haven't been
    # scanned recently (or have never been scanned). The watcher handles
    # inotify-driven updates; this is a fallback for changes the watcher
    # can't see (e.g. external edits to a dataset's config.toml that add
    # new image paths, where no inotify event fires).
    dataset_refresh_interval_seconds: float = 300.0

    # Upload
    max_upload_size_bytes: int = 524_288_000  # ~500 MB

    # HTTP client timeouts
    http_timeout_connect: float = 30.0
    http_timeout_read: float | None = None
    http_timeout_write: float = 30.0
    http_timeout_pool: float = 30.0

    # Cache TTL for the `/api/envs/<name>/models` endpoint and the captioner's
    # `/models` probes. Short enough to pick up newly added models, long
    # enough to keep model pickers snappy.
    api_models_cache_ttl: float = DEFAULT_MODELS_CACHE_TTL_SECONDS

    # Upper bound (seconds) for the entire ``/api/envs/<name>/models``
    # operation — env decryption + API-type inference probes + the
    # per-backend list call. Exceeding this surfaces as HTTP 504
    # GATEWAY_TIMEOUT so a misconfigured/unreachable env fails fast in
    # the WebUI model picker instead of hanging. ``None`` would
    # effectively re-enable indefinite blocking, so this is a
    # ``float`` (no ``None``) on purpose.
    list_models_timeout: float = DEFAULT_LIST_MODELS_TIMEOUT_SECONDS

    # Captioning job cleanup
    captioning_cleanup_interval_seconds: float = 10.0
    captioning_cleanup_grace_seconds: float = 30.0

    # Refine result cache
    refine_result_buffer_size: int = 100

    # yadc paths
    config_path: str = field(default_factory=lambda: str(CONFIG_PATH))
    state_path: str = field(default_factory=lambda: str(STATE_PATH))
    cache_path: str = field(default_factory=lambda: str(CACHE_PATH))

    # Database
    db_path: str = field(default_factory=lambda: str(STATE_PATH / "webui.db"))
