"""``UvicornLoggingConfig`` — sole owner of uvicorn logger configuration.

Mutates ``uvicorn.config.LOGGING_CONFIG`` in place during
``SetupAppEvent`` (fired by ``Application.configure_app()`` *before*
``uvicorn.Config(...)`` is built) so the changes are visible when
``uvicorn.Server.run()`` later calls ``dictConfig``.

Picks the access handler class from ``Configuration.access_log_user_specified``:
``RotatingFileHandler`` for the default cache path, ``WatchedFileHandler``
when the user supplies a path (defers rotation to their external setup).
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path

import uvicorn.config

from yadc.api.configuration import Configuration

from ..events import SetupAppEvent
from .event_dispatcher import event_handler
from .logging_factory import LoggingFactory
from .service import Service


class UvicornLoggingConfig(Service):
    def __init__(self, configuration: Configuration, logging_factory: LoggingFactory):
        self._logger: Logger = logging_factory.get_logger(__name__)
        self._level: int = logging_factory.log_level
        self._log_format: str = configuration.logging_log_format
        self._access_log_file: Path = Path(configuration.access_log_file)
        self._access_log_user_specified: bool = configuration.access_log_user_specified
        self._access_log_max_bytes: int = configuration.access_log_max_bytes
        self._access_log_backup_count: int = configuration.access_log_backup_count

    @event_handler(SetupAppEvent)
    def on_setup_app(self, event: SetupAppEvent):  # pyright: ignore[reportUnusedParameter]
        config = uvicorn.config.LOGGING_CONFIG
        level_name = logging.getLevelName(self._level)

        # Default formatter: yadc format. Access formatter: keep uvicorn's
        # fields but drop the "INFO:    " levelprefix (noise in a file)
        # and force use_colors=False (file sink, not a terminal).
        config["formatters"]["default"]["fmt"] = self._log_format
        config["formatters"]["access"]["fmt"] = '%(client_addr)s - "%(request_line)s" %(status_code)s'
        config["formatters"]["access"]["use_colors"] = False

        self._access_log_file.parent.mkdir(parents=True, exist_ok=True)

        # Default cache path → auto-rotate (no external setup assumed).
        # User-supplied path → watched (defer to user's rotation; don't
        # double-rotate on top of logrotate etc.).
        handler: dict[str, object] = {
            "formatter": "access",
            "filename": str(self._access_log_file),
            "mode": "a",
            "encoding": "utf-8",
        }
        if self._access_log_user_specified:
            handler["class"] = "logging.handlers.WatchedFileHandler"
        else:
            handler["class"] = "logging.handlers.RotatingFileHandler"
            handler["maxBytes"] = self._access_log_max_bytes
            handler["backupCount"] = self._access_log_backup_count
        config["handlers"]["access"] = handler

        # All four uvicorn logger levels — this service is the sole owner
        # (Application.run() does not pass log_level= to uvicorn.Config,
        # which would clobber these post-dictConfig).
        loggers = config["loggers"]
        loggers["uvicorn"]["level"] = level_name
        loggers["uvicorn.error"]["level"] = level_name
        # uvicorn.asgi isn't in the default LOGGING_CONFIG["loggers"] —
        # add the entry so dictConfig creates it with the right level
        # (otherwise it falls through to root WARNING).
        loggers.setdefault("uvicorn.asgi", {})["level"] = level_name
        loggers["uvicorn.access"]["level"] = "INFO"
