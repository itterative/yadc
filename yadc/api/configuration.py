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

from yadc.captioners.api.constants import DEFAULT_LIST_MODELS_TIMEOUT_SECONDS
from yadc.cmd.app import CACHE_PATH, CONFIG_PATH, STATE_PATH
from yadc.llm.constants import DEFAULT_MODELS_CACHE_TTL_SECONDS


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

    # Tagger result cache (LRU; key includes model + threshold fingerprint
    # so config changes naturally evict stale entries).
    tagger_result_buffer_size: int = 500

    # Tagger (ONNX)
    # The tagger is enabled when EITHER a local model path OR a HF
    # repo id is set (see ``TaggingService.is_configured``). With both
    # empty the tagger endpoints return 503.
    # Path to a local ONNX model file. The label file is auto-discovered
    # as ``<model_dir>/selected_tags.csv`` when ``tagger_label_path``
    # is empty. Ignored when ``tagger_repo_id`` is set.
    tagger_model_path: str = ""
    tagger_label_path: str = ""
    # HuggingFace Hub repo to download the model from. When set, the
    # worker downloads ``tagger_repo_model_filename`` and
    # ``tagger_repo_label_filename`` from this repo on startup and
    # uses the cached paths (``tagger_model_path`` /
    # ``tagger_label_path`` are ignored). Defaults match the
    # SmilingWolf / wd-tagger convention.
    tagger_repo_id: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    tagger_repo_model_filename: str = "model.onnx"
    tagger_repo_label_filename: str = "selected_tags.csv"
    # Per-category score thresholds (SmilingWolf / WD defaults).
    # Tags below the threshold for their category are dropped from the
    # response. 0.0 keeps everything in that category.
    tagger_rating_threshold: float = 0.0
    tagger_general_threshold: float = 0.35
    tagger_character_threshold: float = 0.85
    # Turn underscored tag names (``long_hair``) into spaces (``long hair``).
    # Kaomojis are always preserved. Applied post-threshold so only
    # surviving tags are touched. Off by default to preserve raw model
    # output; opt in per-request from the UI.
    tagger_replace_underscores: bool = False
    # Subprocess liveness knobs. The worker pushes a heartbeat while
    # idle (``tagger_heartbeat_interval_seconds``); the main process
    # polls the response queue at that granularity and checks
    # ``is_alive()`` on each timeout so a dead worker is detected
    # within ~one heartbeat. A wedged-but-alive worker is given up
    # after ``tagger_response_timeout_seconds``. Lowering the heartbeat
    # detects death faster at the cost of more idle chatter.
    tagger_heartbeat_interval_seconds: float = 15.0
    tagger_response_timeout_seconds: float = 120.0
    # How often the parent polls the tagger response queue while waiting.
    # Bounds how quickly a dead/killed worker is noticed and how fast a
    # cancel/kill unwinds. Decoupled from the worker heartbeat above.
    tagger_liveness_poll_seconds: float = 1.0
    # Grace window before force-killing the subprocess on cancel. After
    # setting a job's stop_event, cancel waits this long for an in-flight
    # inference to finish on its own; only if it's still running does it
    # terminate the process (ONNX inference can't be interrupted mid-call).
    tagger_cancel_grace_seconds: float = 1.0
    # How long the tagger subprocess is allowed to sit idle (no
    # tagging requests) before the service tears it down. The next
    # request after teardown respawns it. Set to 0 to disable idle
    # teardown (the subprocess stays up forever once started).
    tagger_idle_timeout_seconds: float = 900.0
    # Grace window between a batch job ending and clearing its
    # expected-changes source tag. Residual inotify events from the
    # last writes land slightly after the loop exits (kernel buffering
    # + the watcher debounce); clearing the tag too early lets them
    # leak through re-tagged with the per-file ``source`` ("tagger")
    # instead of the job_id, which the originating tab can't suppress.
    # Mirrors captioning's deferred clear. Must exceed
    # ``watcher_debounce_seconds`` + ``watcher_expected_file_ttl``.
    tagger_expected_changes_grace_seconds: float = 5.0

    # yadc paths
    config_path: str = field(default_factory=lambda: str(CONFIG_PATH))
    state_path: str = field(default_factory=lambda: str(STATE_PATH))
    cache_path: str = field(default_factory=lambda: str(CACHE_PATH))

    # Database
    db_path: str = field(default_factory=lambda: str(STATE_PATH / "webui.db"))
