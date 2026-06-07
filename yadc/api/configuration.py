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
- **HTTP client timeouts** — connect/read/write/pool, threaded into
  ``AsyncSession``.
- **Caches** — ``api_models_cache_ttl`` for ``/api/envs/.../models``,
  ``refine_result_buffer_size`` for the refine endpoint.
- **Yadc paths** — ``config_path``, ``state_path``, ``cache_path``,
  ``db_path`` (SQLite for the web UI's datasets/configs/settings tables).
"""

from dataclasses import dataclass, field

from yadc.captioners.api.constants import DEFAULT_MODELS_CACHE_TTL_SECONDS
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

    # SSE
    sse_listeners_warning: int = 25
    sse_listeners_max: int = 100
    sse_listener_max_events: int = 100
    sse_event_history_size: int = 128

    # Server
    graceful_shutdown_timeout: int = 5

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
