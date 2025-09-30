import sys

import click

from yadc.core import logging
from yadc.cmd import status as cmd_status, configs as cmd_configs

from . import cli_common

_logger = logging.get_logger(__name__)

@click.group(
    'configs',
    short_help='Manage the user configs',
    help='Manage the user configs'
)
def configs():
    pass

@configs.command(
    'list',
    short_help='List available user configs',
    help='List available user configs',
)
@cli_common.log_level
def list():
    configs = cmd_configs.list_user_config()

    if not configs:
        _logger.warning('No user configs found.')
        return

    for template in configs:
        click.echo(template)

@configs.command(
    'add',
    short_help='Add a new user config',
    help='Add a new user config',
)
@click.argument('name', type=str)
@cli_common.log_level
def add(name: str):
    try:
        cmd_configs.load_user_config(name)

        _logger.error('Error: config with name "%s" already exists. Use "configs edit" to edit it.', name)
        sys.exit(cmd_status.STATUS_USER_ERROR)
    except FileNotFoundError:
        pass

    default_template = '# Edit the template below\n\n'

    user_template = click.edit(default_template, extension='.toml', require_save=True)

    if user_template is None:
        _logger.info('Cancelled.')
        return

    try:
        cmd_configs.save_user_config(name, user_template)
    except PermissionError:
        _logger.error('Error: failed to save user config: permissions error')
        sys.exit(cmd_status.STATUS_ERROR)

    _logger.info('Added new user config: %s', name)

@configs.command(
    'delete',
    short_help='Delete a user config',
    help='Delete an existing user config',
)
@click.argument('name', type=str)
@cli_common.log_level
def delete(name: str):
    if not cmd_configs.delete_user_config(name):
        _logger.warning('Warning: config with name "%s" dot not exist.', name)
        return

@configs.command(
    'edit',
    short_help='Edit a user config',
    help='Edit an existing user config',
)
@click.argument('name', type=str)
@cli_common.log_level
def edit(name: str):
    try:
        user_template = cmd_configs.load_user_config(name)
    except FileNotFoundError:
        _logger.error('Error: template with name "%s" does not exit. Use "configs add" to add it.', name)
        sys.exit(cmd_status.STATUS_USER_ERROR)


    user_template = click.edit(user_template, extension='.toml', require_save=True)

    if user_template is None:
        _logger.info('Cancelled.')
        return

    try:
        cmd_configs.save_user_config(name, user_template)
    except PermissionError:
        _logger.error('Error: failed to save user config: permissions error')
        sys.exit(cmd_status.STATUS_ERROR)

    _logger.info('Updated user config: %s', name)
