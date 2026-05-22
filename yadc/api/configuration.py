from dataclasses import dataclass, field

from yadc.cmd.app import CACHE_PATH, CONFIG_PATH, STATE_PATH


@dataclass
class Configuration:
    # HTTP
    http_host: str = "127.0.0.1"
    http_port: int = 7860
    http_threads: int = 4

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

    # yadc paths
    config_path: str = field(default_factory=lambda: str(CONFIG_PATH))
    state_path: str = field(default_factory=lambda: str(STATE_PATH))
    cache_path: str = field(default_factory=lambda: str(CACHE_PATH))

    # Database
    db_path: str = field(default_factory=lambda: str(STATE_PATH / "webui.db"))
