"""``UvicornLoggingConfig`` — reconfigures uvicorn's loggers to use the yadc format and level.

uvicorn ships with its own ``LOGGING_CONFIG`` dict that installs a different
formatter (``%(levelprefix)s %(message)s``) on the ``uvicorn``,
``uvicorn.error``, and ``uvicorn.access`` loggers, and a default level
(``INFO``) that is not affected by the yadc ``--log-level`` flag. This
service mutates ``uvicorn.config.LOGGING_CONFIG`` in place during
``SetupAppEvent`` — fired by ``Application.configure_app()`` *before*
``uvicorn.Config(...)`` is built in ``Application.run()`` — so the changes
are visible when ``uvicorn.Server.run()`` later calls
``logging.config.dictConfig``.

The yadc log level comes from ``LoggingFactory`` (read by
``Application.run()`` and passed to ``uvicorn.Config(log_level=...)``).
uvicorn applies ``Config.log_level`` to the ``uvicorn.error`` /
``uvicorn.access`` / ``uvicorn.asgi`` loggers *after* ``dictConfig`` runs,
so this service also mutates the level for the main ``uvicorn`` logger in
``LOGGING_CONFIG`` (``Config.log_level`` does not touch that one).
"""

from __future__ import annotations

import logging
from logging import Logger

import uvicorn.config

from yadc.api.configuration import Configuration

from ..events import SetupAppEvent
from .event_dispatcher import event_handler
from .logging_factory import LoggingFactory
from .service import Service


class UvicornLoggingConfig(Service):
    """Mutates ``uvicorn.config.LOGGING_CONFIG`` so uvicorn's log lines
    use the yadc format and respect ``LoggingFactory.log_level``."""

    def __init__(self, configuration: Configuration, logging_factory: LoggingFactory):
        self._logger: Logger = logging_factory.get_logger(__name__)
        self._level: int = logging_factory.log_level
        self._log_format: str = configuration.logging_log_format

    @event_handler(SetupAppEvent)
    def on_setup_app(self, event: SetupAppEvent):  # pyright: ignore[reportUnusedParameter]
        config = uvicorn.config.LOGGING_CONFIG

        # Default formatter handles uvicorn's general log lines;
        # access formatter handles per-request access log lines.
        # Both get the yadc format so they match the rest of the API.
        config["formatters"]["default"]["fmt"] = self._log_format
        config["formatters"]["access"]["fmt"] = self._log_format

        # Main ``uvicorn`` logger's level is set here (not by Config.log_level).
        # The other uvicorn loggers get their level from Config.log_level,
        # which Application.run() sources from LoggingFactory.log_level.
        level_name = logging.getLevelName(self._level)
        config["loggers"]["uvicorn"]["level"] = level_name
