import click

def log_level(f):
    def set_log_level(ctx, param, level: str):
        from yadc.core import logging
        logging.set_level(level)

    return click.option(
        '--log-level',
        default='info',
        type=click.Choice(['info', 'warning', 'error', 'debug']),
        help='Set the logging level',
        expose_value=False,
        callback=set_log_level,
    )(f)

def env(f):
    return click.option(
        '--env',
        type=str,
        default='default',
        help='Configuration environment',
    )(f)
