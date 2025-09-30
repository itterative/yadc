import sys

from importlib import resources

import click

import yadc.templates

from yadc.core import logging, env
from yadc.cmd import status

from . import cli_common

APP_NAME= 'yadc'
TEMPLATE_PATH = env.STATE_PATH / 'templates'
TEMPLATE_PATH.mkdir(mode=0o750, exist_ok=True)

_logger = logging.get_logger(__name__)

@click.group(
    'templates',
    short_help='Manage the user templates',
    help='Manage the user templates'
)
def templates():
    pass

@templates.command(
    'list',
    short_help='List available user templates',
    help='List available user templates',
)
@cli_common.log_level
def list():
    templates = yadc.templates.list_user_template()

    if not templates:
        _logger.warning('No user templates found.')
        return

    for template in templates:
        click.echo(template)

@templates.command(
    'add',
    short_help='Add a new user template',
    help='Add a new user template',
)
@click.argument('name', type=str)
@cli_common.log_level
def add(name: str):
    try:
        yadc.templates.load_user_template(name)

        _logger.error('Error: template with name "%s" already exists. Use "templates edit" to edit it.', name)
        sys.exit(status.STATUS_USER_ERROR)
    except FileNotFoundError:
        pass

    default_template = '# Edit the template below\n\n' + yadc.templates.default_template()

    user_template = click.edit(default_template, extension='.jinja', require_save=True)

    if user_template is None:
        _logger.info('Cancelled.')
        return

    try:
        yadc.templates.save_user_template(name, user_template)
    except PermissionError:
        _logger.error('Error: failed to save user template: permissions error')
        sys.exit(status.STATUS_ERROR)

    _logger.info('Added new user template: %s', name)

@templates.command(
    'delete',
    short_help='Delete a user template',
    help='Delete an existing user template',
)
@click.argument('name', type=str)
@cli_common.log_level
def delete(name: str):
    if not yadc.templates.delete_user_template(name):
        _logger.warning('Warning: template with name "%s" dot not exist.', name)
        return

@templates.command(
    'edit',
    short_help='Edit a user template',
    help='Edit an existing user template',
)
@click.argument('name', type=str)
@cli_common.log_level
def edit(name: str):
    try:
        user_template = yadc.templates.load_user_template(name)
    except FileNotFoundError:
        _logger.error('Error: template with name "%s" does not exit. Use "templates add" to add it.', name)
        sys.exit(status.STATUS_USER_ERROR)


    user_template = click.edit(user_template, extension='.jinja', require_save=True)

    if user_template is None:
        _logger.info('Cancelled.')
        return

    try:
        yadc.templates.save_user_template(name, user_template)
    except PermissionError:
        _logger.error('Error: failed to save user template: permissions error')
        sys.exit(status.STATUS_ERROR)

    _logger.info('Updated user template: %s', name)
