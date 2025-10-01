import logging
import click

class ClickHandler(logging.Handler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.styles: dict[str, dict] = {
            'info': dict(fg='green'),
            'error': dict(fg='red'),
            'exception': dict(fg='red'),
            'critical': dict(fg='red'),
            'debug': dict(fg='blue'),
            'trace': dict(fg='bright_blue'),
            'warning': dict(fg='yellow')
        }

    def emit(self, record: logging.LogRecord):
        level = record.levelname.lower()
        msg = self.format(record)

        if kwargs := self.styles.get(level, None):
            click.secho(msg, **kwargs)
        else:
            click.echo(msg)
