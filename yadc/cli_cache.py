import click

from yadc.cmd import cache as cmd_cache
from yadc.core import logging

from . import cli_common

_logger = logging.get_logger(__name__)


@click.group(
    "cache",
    short_help="Manage the cache directory",
    help="Manage the yadc cache directory, which stores API request caches and debug logs.",
)
def cache():
    pass


@cache.command(
    "dir",
    short_help="Show the cache directory",
    help="Show the path to the cache directory",
)
@cli_common.log_level
def dir_cmd():
    click.echo(cmd_cache.get_cache_dir())


@cache.command(
    "clean",
    short_help="Remove all cached data",
    help="Remove the entire cache directory, including API request caches and debug logs",
)
@cli_common.log_level
def clean():
    bytes_freed = cmd_cache.clean_cache()
    if bytes_freed == 0:
        _logger.info("Cache directory does not exist, nothing to clean.")
        return

    _logger.info("Cleaned cache (%s freed).", _format_size(bytes_freed))


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MiB"
    else:
        return f"{size / (1024 * 1024 * 1024):.1f} GiB"
