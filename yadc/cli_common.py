"""Shared click options (``--log-level``, ``--env``) reused across CLI commands."""

from collections.abc import Callable
from typing import Any

import click


def log_level(f: Callable[..., Any]) -> Callable[..., Any]:
    def set_log_level(_ctx: Any, _param: Any, level: str) -> None:
        from yadc.core import logging

        logging.set_level(level)

    return click.option(
        "--log-level",
        default="info",
        type=click.Choice(["info", "warning", "error", "debug", "trace"]),
        help="Set the logging level",
        expose_value=False,
        callback=set_log_level,
    )(f)


def env(f: Callable[..., Any]) -> Callable[..., Any]:
    return click.option(
        "--env",
        type=str,
        default="default",
        help="Configuration environment",
    )(f)
