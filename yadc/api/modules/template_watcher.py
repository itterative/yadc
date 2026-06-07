"""Filesystem watcher for template changes.

Watches the user templates directory for changes to ``*.jinja`` files and
emits ``TemplatesChangedEvent`` via SSE so all connected clients stay in sync.
"""

from __future__ import annotations

from typing import override

from watchdog.observers.api import BaseObserver

from yadc.cmd import templates as cmd_templates
from yadc.cmd.app import STATE_PATH

from ..events import Event, TemplatesChangedEvent
from ..watcher_base import SinglePathWatcherService
from .event_dispatcher import EventDispatcher
from .logging_factory import LoggingFactory
from .thread_factory import ThreadFactory

_TEMPLATE_DIR = str(STATE_PATH / "templates")
_TEMPLATE_EXT = ".jinja"


def _is_template(path: str) -> bool:
    return path.endswith(_TEMPLATE_EXT)


class TemplateWatcherService(SinglePathWatcherService):
    """Watches the templates directory for changes and emits SSE events."""

    def __init__(
        self,
        event_dispatcher: EventDispatcher,
        logging: LoggingFactory,
        observer: BaseObserver,
        thread_factory: ThreadFactory,
    ) -> None:
        super().__init__(
            event_dispatcher,
            logging,
            watch_path=_TEMPLATE_DIR,
            file_filter=_is_template,
            watch_label="Template",
            observer=observer,
            thread_factory=thread_factory,
        )

    @override
    def create_event(self, changed_files: frozenset[str]) -> Event:
        return TemplatesChangedEvent(templates=cmd_templates.list_user_template())
