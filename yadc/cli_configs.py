import sys

import click

from yadc.cmd import configs as cmd_configs
from yadc.cmd import status as cmd_status
from yadc.core import logging

from . import cli_common

_logger = logging.get_logger(__name__)


@click.group(
    "configs",
    short_help="Manage the user configs",
    help="User configs allow you to reuse certain settings across multiple datasets, for example, "
    "when you want to have specific settings applied to one model.",
)
def configs():
    pass


@configs.command(
    "dir",
    short_help="Show the storage directory",
    help="Show the directory where user configs are stored",
)
@cli_common.log_level
def path():
    click.echo(cmd_configs.configs.CONFIG_PATH)


@configs.command(
    "list",
    short_help="List available user configs",
    help="List available user configs",
)
@cli_common.log_level
def list():
    configs = cmd_configs.list_user_config()

    if not configs:
        _logger.warning("No user configs found.")
        return

    for template in configs:
        click.echo(template)


@configs.command(
    "delete",
    short_help="Delete a user config",
    help="Delete an existing user config",
)
@click.argument("name", type=str)
@cli_common.log_level
def delete(name: str):
    if not cmd_configs.delete_user_config(name):
        _logger.warning('Warning: config with name "%s" dot not exist.', name)
        return


@configs.command(
    "edit",
    short_help="Edit a user config",
    help="Edit an existing user config, or create a new one if it does not exist",
)
@click.argument("name", type=str)
@cli_common.log_level
def edit(name: str):
    is_new = False
    try:
        user_template = cmd_configs.load_user_config(name)
    except FileNotFoundError:
        is_new = True
        user_template = "# Edit the template below\n\n"

    user_template = click.edit(user_template, extension=".toml", require_save=True)

    if user_template is None:
        _logger.info("Cancelled.")
        return

    try:
        cmd_configs.save_user_config(name, user_template)
    except PermissionError:
        _logger.error("Error: failed to save user config: permissions error")
        sys.exit(cmd_status.STATUS_ERROR)

    if is_new:
        _logger.info("Created new user config: %s", name)
    else:
        _logger.info("Updated user config: %s", name)
