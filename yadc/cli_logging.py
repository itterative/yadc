import logging
from typing import Any

import click
from typing_extensions import override


class ClickHandler(logging.Handler):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.styles: dict[str, dict[str, Any]] = {
            "info": dict(fg="green"),
            "error": dict(fg="red"),
            "exception": dict(fg="red"),
            "critical": dict(fg="red"),
            "debug": dict(fg="blue"),
            "trace": dict(fg="bright_blue"),
            "warning": dict(fg="yellow"),
        }

    @override
    def emit(self, record: logging.LogRecord):
        level = record.levelname.lower()
        msg = self.format(record)

        if kwargs := self.styles.get(level, None):
            click.secho(msg, err=True, **kwargs)
        else:
            click.echo(msg, err=True)
