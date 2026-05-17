import sys

import click

from yadc.cmd import status as cmd_status
from yadc.cmd import templates as cmd_templates
from yadc.core import logging

from . import cli_common

_logger = logging.get_logger(__name__)


@click.group("templates", short_help="Manage the user templates", help="User templates allow you to reuse certain prompt templates across multiple datasets.")
def templates():
    pass


@templates.command(
    "list",
    short_help="List available user templates",
    help="List available user templates",
)
@cli_common.log_level
def list():
    templates = cmd_templates.list_user_template()

    if not templates:
        _logger.warning("No user templates found.")
        return

    for template in templates:
        click.echo(template)


@templates.command(
    "delete",
    short_help="Delete a user template",
    help="Delete an existing user template",
)
@click.argument("name", type=str)
@cli_common.log_level
def delete(name: str):
    if not cmd_templates.delete_user_template(name):
        _logger.warning('Warning: template with name "%s" dot not exist.', name)
        return


@templates.command(
    "edit",
    short_help="Edit a user template",
    help="Edit an existing user template, or create a new one if it does not exist",
)
@click.argument("name", type=str)
@cli_common.log_level
def edit(name: str):
    is_new = False
    try:
        user_template = cmd_templates.load_user_template(name)
    except FileNotFoundError:
        is_new = True
        user_template = "# Edit the template below\n\n" + cmd_templates.default_template()

    user_template = click.edit(user_template, extension=".jinja", require_save=True)

    if user_template is None:
        _logger.info("Cancelled.")
        return

    try:
        cmd_templates.save_user_template(name, user_template)
    except PermissionError:
        _logger.error("Error: failed to save user template: permissions error")
        sys.exit(cmd_status.STATUS_ERROR)

    if is_new:
        _logger.info("Created new user template: %s", name)
    else:
        _logger.info("Updated user template: %s", name)
