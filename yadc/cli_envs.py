"""``yadc envs`` — manage API environments (URL, token, model).

Environments are named entries in the user config TOML that bundle an
API URL, token, and model name so a single dataset config can switch
between backends without re-entering credentials. Tokens are
encrypted-at-rest using the active ``KeyStorage`` backend (keyring or
password-derived AES), so this command also surfaces the
``encrypt`` / ``decrypt`` / ``change-password`` / ``reset-encryption``
subcommands that flip between encryption modes and rotate the key.
"""

import sys
from typing import Literal, cast

import click
import pydantic

from yadc.cmd import app as cmd_app
from yadc.cmd import config as cmd_config
from yadc.cmd import envs as cmd_envs
from yadc.cmd import status as cmd_status
from yadc.cmd.envs.keystorage_password import PasswordRequiredError
from yadc.core import logging
from yadc.core.env import YADC_PASSWORD

from . import cli_common

_logger = logging.get_logger(__name__)


@click.group(
    "envs",
    short_help="Manage the user environments",
    help="User environments are used in order to store API details in a safe manner. "
    "They allow you to reuse APIs over different captioning datasets without having to add them in the dataset TOML file.",
)
def envs():
    pass


@envs.group(
    "key-mode",
    short_help="Manage key storage mode",
    help="Switch between keyring and password-based private key storage.",
)
def key_mode():
    pass


@key_mode.command("get", short_help="Show current key storage mode")
@cli_common.log_level
def key_mode_get():
    try:
        config = cmd_config.load_config()
        click.echo(config.key_storage.mode)
    except Exception as e:
        _logger.error("Error: failed to read key storage mode: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)


def _read_password(prompt: str) -> str | None:
    """Read password from YADC_PASSWORD env, stdin, or interactive prompt."""
    env_password = YADC_PASSWORD
    if env_password:
        return env_password
    if not sys.stdin.isatty():
        password = sys.stdin.read().strip()
        return password if password else None
    return click.prompt(prompt, hide_input=True)


@key_mode.command("set", short_help="Set key storage mode")
@click.argument("mode", type=click.Choice(["keyring", "password"]))
@cli_common.log_level
def key_mode_set(mode: str):
    try:
        config = cmd_config.load_config()
    except Exception as e:
        _logger.error("Error: failed to read user config: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    password: str | None = None
    old_password: str | None = None

    if config.key_storage.mode == "password" and mode == "keyring":
        old_password = _read_password("Enter current password to decrypt existing tokens")

    if mode == "password":
        password = _read_password("Enter new password for key storage")

    try:
        cmd_envs.set_key_mode(
            config,
            cast(Literal["keyring", "password"], mode),
            password=password,
            old_password=old_password,
        )
        _logger.info("Key storage mode set to '%s'.", mode)
    except Exception as e:
        _logger.error("Error: failed to set key storage mode: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)


@envs.command(
    "dir",
    short_help="Show the storage directory",
    help="Show the directory where user environments are stored",
)
@cli_common.log_level
def envs_path():
    click.echo(cmd_app.CONFIG_PATH)


@envs.command(
    "get",
    short_help="Retrieve a setting",
    help="Retrieve a setting value from user environments",
)
@click.argument("key", type=click.Choice(list(cmd_config.AppConfigEnv.model_fields.keys())))
@cli_common.log_level
@cli_common.env
def envs_get(key: str, env: str = "default"):
    if key not in cmd_config.AppConfigEnv.model_fields:
        _logger.error("Error: invalid setting: %s", key)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    try:
        env_config = cmd_envs.get_env(env) or cmd_config.AppConfigEnv()
        value_obj = getattr(env_config, key)
        if value_obj.value is None:
            _logger.error("Error: key not found: %s (env: %s)", key, env)
            sys.exit(cmd_status.STATUS_USER_ERROR)

        if value_obj.method == "none":
            click.echo(value_obj.value)
            return

        method = cmd_envs.EncryptionMethod(value_obj.method)
        try:
            decrypted = cmd_envs.decrypt_setting(value_obj.value, method=method)
        except PasswordRequiredError:
            password = _read_password("Enter password to decrypt this setting")
            if password is None:
                _logger.error("Error: password required to decrypt '%s'.", key)
                sys.exit(cmd_status.STATUS_USER_ERROR)
            decrypted = cmd_envs.decrypt_setting(value_obj.value, method=method, password=password)

        if decrypted is not None:
            click.echo(decrypted)
            return

        _logger.error("Error: failed to decrypted setting.")
        sys.exit(cmd_status.STATUS_ERROR)
    except Exception as e:
        _logger.error("Error: failed to read user environment: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)


@envs.command(
    "set",
    short_help="Update a setting",
    help="Update a setting value in user environments",
)
@click.argument("key", type=click.Choice(list(cmd_config.AppConfigEnv.model_fields.keys())))
@click.argument("value", type=str, required=False, default=None)
@click.option("--force", is_flag=True, help="Recreates the user environment if invalid")
@cli_common.log_level
@cli_common.env
def envs_set(key: str, value: str | None, env: str = "default", force: bool = False):
    if key not in cmd_config.AppConfigEnv.model_fields:
        _logger.error("Error: invalid setting: %s", key)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    if value is None:
        secret_value = key in cmd_config.AppConfigEnv.ENCRYPTED_FIELDS
        value = click.prompt(f"Enter {key}", hide_input=secret_value)

    try:
        config = cmd_config.load_config()
    except (pydantic.ValidationError, ValueError):
        if not force:
            _logger.error("Error: user environments is invalid")
            sys.exit(cmd_status.STATUS_USER_ERROR)
        config = cmd_config.AppConfig(key_storage=cmd_config.AppConfigKeyStorage(mode="keyring"), envs={})
    except Exception as e:
        _logger.error("Error: failed to read user environments: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    try:
        cmd_envs.update_env(key, value, env=env, config=config)
    except ValueError as e:
        _logger.error("Error: %s", e)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    try:
        cmd_envs.save_env(config=config)
        _logger.info("User environment setting %s (env: %s) has been updated.", key, env)
    except Exception as e:
        _logger.error("Error: user environments could not be updated: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)


@envs.command("delete", short_help="Delete a setting", help="Delete a setting from user environments")
@click.argument("key", type=click.Choice(list(cmd_config.AppConfigEnv.model_fields.keys())))
@click.option("--force", is_flag=True, help="Recreates the user environment if invalid")
@cli_common.log_level
@cli_common.env
def envs_delete(key: str, env: str = "default", force: bool = False):
    if key not in cmd_config.AppConfigEnv.model_fields:
        _logger.error("Error: invalid setting: %s", key)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    try:
        config = cmd_config.load_config()
    except (pydantic.ValidationError, ValueError):
        if not force:
            _logger.error("Error: user environments is invalid")
            sys.exit(cmd_status.STATUS_USER_ERROR)
        config = cmd_config.AppConfig(key_storage=cmd_config.AppConfigKeyStorage(mode="keyring"), envs={})
    except Exception as e:
        _logger.error("Error: failed to read user environments: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    try:
        cmd_envs.update_env(key, None, env=env, config=config)
    except ValueError as e:
        _logger.error("Error: %s", e)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    try:
        cmd_envs.save_env(config=config)
        _logger.info("User environment setting %s (env: %s) has been removed.", key, env)
    except Exception as e:
        _logger.error("Error: user environments could not be updated: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)


@envs.command("clear", short_help="Clear the user environment", help="Clear the user environment. You can either clear one environment or all.")
@click.option("--all", is_flag=True, help="Clears all environments (including encryption keys)")
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


@envs.command("list", short_help="List available envs", help="List available envs in user environments")
@cli_common.log_level
def envs_list():
    try:
        available_envs = cmd_envs.list_all_env()
    except Exception as e:
        _logger.error("Error: failed to read user environments: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    if not available_envs:
        _logger.warning("No environment found in user environments.")
        sys.exit(cmd_status.STATUS_OK)

    for env in available_envs:
        click.echo(env)


@envs.command(
    "show",
    short_help="List all settings",
    help='List all settings in user environments. Secret settings are redacted. If you want to view the values for secret settings, use "config get SETTING"',
)
@cli_common.log_level
@cli_common.env
def envs_show(env: str = "default"):
    try:
        env_config = cmd_envs.get_env(env) or cmd_config.AppConfigEnv()
    except Exception as e:
        _logger.error("Error: failed to read user environments: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    found = False
    buffer = ""

    for key in cmd_config.AppConfigEnv.model_fields:
        value_obj = getattr(env_config, key)
        assert isinstance(value_obj, cmd_config.AppConfigEnvValue)
        if value_obj.value is None:
            continue

        display = "[REDACTED]" if value_obj.is_encrypted else value_obj.value
        buffer += f"{key} = {display}\n"
        found = True

    if not found:
        _logger.warning("No user environments values found.")
        sys.exit(cmd_status.STATUS_OK)

    click.echo(buffer.strip())
