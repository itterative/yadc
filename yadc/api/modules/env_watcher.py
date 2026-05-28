"""Filesystem watcher for environment config changes.

Watches the user config directory for changes to ``config.toml`` and emits
``EnvironmentsChangedEvent`` via SSE so all connected clients stay in sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from yadc.cmd import envs as cmd_envs
from yadc.cmd.app import CONFIG_PATH

from ..events import EnvironmentsChangedEvent, Event
from ..watcher_base import SinglePathWatcherService
from .event_dispatcher import EventDispatcher
from .logging_factory import LoggingFactory

_CONFIG_FILE = "config.toml"


def _is_config_toml(path: str) -> bool:
    return Path(path).name == _CONFIG_FILE


class EnvWatcherService(SinglePathWatcherService):
    """Watches ``config.toml`` for changes and emits SSE events."""

    def __init__(self, event_dispatcher: EventDispatcher, logging: LoggingFactory) -> None:
        super().__init__(
            event_dispatcher,
            logging,
            watch_path=str(CONFIG_PATH),
            file_filter=_is_config_toml,
            watch_label="Env config",
        )

    @override
    def create_event(self, changed_files: frozenset[str]) -> Event:
        # All envs live in a single config.toml — report all names since
        # we cannot determine which specific envs changed from the
        # filesystem event alone.
        return EnvironmentsChangedEvent(envs=cmd_envs.list_all_env())
