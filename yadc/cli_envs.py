from typing import Optional

import sys
import pydantic

import click
from . import cli_common

from yadc.core import logging
from yadc.cmd import app as cmd_app, status as cmd_status, envs as cmd_envs

_logger = logging.get_logger(__name__)

@click.group(
    'envs',
    short_help='Manage the user environments',
    help='User environments are used in order to store API details in a safe manner. They allow you to reuse APIs over different captioning datasets without having to add them in the dataset TOML file.',
)
def envs():
    pass

@envs.command(
    'get',
    short_help='Retrieve a setting',
    help='Retrieve a setting value from user environments',
)
@click.argument('key', type=click.Choice(cmd_envs.ENV_KEYS))
@cli_common.log_level
@cli_common.env
def envs_get(key: str, env: str = "default"):
    if key not in cmd_envs.ENV_KEYS:
        _logger.error('Error: invalid setting: %s', key)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    try:
        env_config = cmd_envs.get_env(env)
        env_setting = env_config[key]

        if not env_setting or not env_setting.value:
            _logger.error('Error: key not found: %s (env: %s)', key, env)
            sys.exit(cmd_status.STATUS_USER_ERROR)

        if not env_setting.encrypted:
            click.echo(env_setting)
            return

        env_setting_decrypted = cmd_envs.decrypt_setting(env_setting.value)
        if env_setting_decrypted is not None:
            click.echo(env_setting_decrypted)
            return

        _logger.error("Error: failed to decrypted setting.")
        sys.exit(cmd_status.STATUS_ERROR)
    except Exception as e:
        _logger.error('Error: failed to read user environment: %s', e)
        sys.exit(cmd_status.STATUS_ERROR)


@envs.command(
    'set',
    short_help='Update a setting',
    help='Update a setting value in user environments',
)
@click.argument('key', type=click.Choice(cmd_envs.ENV_KEYS))
@click.argument('value', type=str, required=False, default=None)
@click.option('--force', is_flag=True, help='Recreates the user environment if invalid')
@cli_common.log_level
@cli_common.env
def envs_set(key: str, value: Optional[str], env: str = "default", force: bool = False):
    if key not in cmd_envs.ENV_KEYS:
        _logger.error('Error: invalid setting: %s', key)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    if value is None:
        secret_value = key in cmd_envs.ENCRYPTED_KEYS
        value = click.prompt(f'Enter {key}', hide_input=secret_value)

    try:
        config_toml = cmd_app.load_config()
    except (pydantic.ValidationError, ValueError):
        if not force:
            _logger.error('Error: user environments is invalid')
            sys.exit(cmd_status.STATUS_ERROR)
        config_toml = {}
    except Exception as e:
        _logger.error('Error: failed to read user environments: %s', e)
        sys.exit(cmd_status.STATUS_ERROR)

    try:
        env_config = cmd_envs.update_env(key, value, env=env, config_toml=config_toml)
    except ValueError as e:
        _logger.error('Error: %s', e)
        sys.exit(cmd_status.STATUS_ERROR)

    try:
        cmd_envs.save_env(env=env, env_config=env_config, config_toml=config_toml)
        _logger.info("User environment setting %s (env: %s) has been updated.", key, env)
    except Exception as e:
        _logger.error("Error: user environments could not be updated: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)


@envs.command(
    'delete',
    short_help='Delete a setting',
    help='Delete a setting from user environments'
)
@click.argument('key', type=click.Choice(cmd_envs.ENV_KEYS))
@click.option('--force', is_flag=True, help='Recreates the user environment if invalid')
@cli_common.log_level
@cli_common.env
def envs_delete(key: str, env: str = "default", force: bool = False):
    if key not in cmd_envs.ENV_KEYS:
        _logger.error('Error: invalid setting: %s', key)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    try:
        config_toml = cmd_app.load_config()
    except (pydantic.ValidationError, ValueError):
        if not force:
            _logger.error('Error: user environments is invalid')
            sys.exit(cmd_status.STATUS_ERROR)
        config_toml = {}
    except Exception as e:
        _logger.error('Error: failed to read user environments: %s', e)
        sys.exit(cmd_status.STATUS_ERROR)

    try:
        env_config = cmd_envs.update_env(key, None, env=env, config_toml=config_toml)
    except ValueError as e:
        _logger.error('Error: %s', e)
        sys.exit(cmd_status.STATUS_ERROR)

    try:
        cmd_envs.save_env(env=env, env_config=env_config, config_toml=config_toml)
        _logger.info("User environment setting %s (env: %s) has been removed.", key, env)
    except Exception as e:
        _logger.error("Error: user environments could not be updated: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)


@envs.command(
    'clear',
    short_help='Clear the user environment',
    help='Clear the user environment. You can either clear one environment or all.'
)
@click.option('--all', is_flag=True, help='Clears all environments (including encryption keys)')
@cli_common.log_level
@cli_common.env
def envs_clear(env: str = "default", all: bool = False):
    if all:
        try:
            cmd_envs.reset_envs()
        except PermissionError:
            _logger.error("Error: user environments could not be cleared: permission denied")
            sys.exit(cmd_status.STATUS_ERROR)
        except Exception as e:
            _logger.error("Error: user environments could not be cleared: %s", e)
            sys.exit(cmd_status.STATUS_ERROR)

        _logger.info("User config cleared.")
        sys.exit(cmd_status.STATUS_OK)

    try:
        cmd_envs.delete_env(env)
        _logger.info("User config cleared. (env: %s)", env)
    except PermissionError:
        _logger.error("Error: user environments could not be cleared: permission denied")
        sys.exit(cmd_status.STATUS_ERROR)
    except Exception as e:
        _logger.error("Error: user environments could not be cleared: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)


@envs.command(
    'list',
    short_help='List available envs',
    help='List available envs in user environments'
)
@cli_common.log_level
def envs_list():
    try:
        available_envs = cmd_envs.list_all_env()
    except Exception as e:
        _logger.error('Error: failed to read user environments: %s', e)
        sys.exit(cmd_status.STATUS_ERROR)

    if not available_envs:
        _logger.warning("No environment found in user environments.")
        sys.exit(cmd_status.STATUS_OK)

    for env in available_envs:
        click.echo(env)


@envs.command(
    'show',
    short_help='List all settings',
    help='List all settings in user environments. Secret settings are redacted. If you want to view the values for secret settings, use "config get SETTING"'
)
@cli_common.log_level
@cli_common.env
def envs_show(env: str = "default"):
    try:
        env_config = cmd_envs.get_env(env)
    except Exception as e:
        _logger.error('Error: failed to read user environments: %s', e)
        sys.exit(cmd_status.STATUS_ERROR)

    found = False

    # buffer so error and warnings show at the beginning
    buffer = ''

    for key, setting in env_config.items():
        if not setting or not setting.value:
            continue

        buffer += f'{key} = {setting}\n'
        found = True

    if not found:
        _logger.warning("No user environments values found.")
        sys.exit(cmd_status.STATUS_OK)

    click.echo(buffer.strip())
