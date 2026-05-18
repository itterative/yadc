import os
from typing import Any, cast

import pydantic
import toml

from yadc.cmd import app
from yadc.core import logging
from yadc.core.user_config import UserConfig, UserConfigApi

from .encryption import decrypt_setting, encrypt_setting
from .setting import Setting

CONFIG_NAME = "config.toml"

ENV_KEYS = [
    "api_url",
    "api_token",
    "api_model_name",
]
ENCRYPTED_KEYS = ["api_token"]

_logger = logging.get_logger(__name__)


def list_all_env() -> list[str]:
    config_toml = app.load_config()
    return list(config_toml.get("env", {}).keys())


def reset_envs():
    config_toml = app.load_config()
    config_toml.pop("env", None)

    _save_config_raw(config_toml)


def load_env(env: str = "default") -> UserConfig:
    try:
        config_toml = app.load_config()
        default_env_config = get_env("default", config_toml=config_toml)
        env_config = get_env(env, config_toml=config_toml)

        config: dict[str, Any] = {}
        for key in set(default_env_config.keys()) | set(env_config.keys()):
            setting = env_config.get(key, None)
            setting = setting or default_env_config.get(key, None)

            if not setting or not setting.value:
                continue

            if key.endswith("_encrypted"):
                # old encrypted setting handling
                key = key.removesuffix("_encrypted")
                setting = Setting(value=decrypt_setting(setting.value), encrypted=False)
            elif setting.encrypted:
                key = key.removesuffix("_encrypted")
                setting = Setting(value=decrypt_setting(setting.value), encrypted=False)

            if setting.encrypted and setting.value is None:
                _logger.warning("Warning: Could not decrypt %s for env '%s'", key, env)
                continue

            config[key] = setting.value

        return UserConfig(
            api=UserConfigApi(
                url=cast(str, config.get("api_url", "")),
                token=cast(str, config.get("api_token", "")),
                model_name=cast(str, config.get("api_model_name", "")),
            ),
        )
    except (pydantic.ValidationError, ValueError):
        _logger.warning("Warning: user config is invalid")
    except PermissionError:
        _logger.warning("Warning: failed to read user config: permission denied")
    except Exception as e:
        _logger.warning("Warning: failed to read user config: %s", e)

    return UserConfig(api=UserConfigApi())


def _save_config_raw(config: dict[str, Any]):
    config_path = app.CONFIG_PATH / CONFIG_NAME

    with open(config_path, "w") as f:
        toml.dump(config, f)

    try:
        os.chmod(config_path, 0o600)
    except Exception as e:
        _logger.warning("Warning: failed to restrict permissions on user config: %s", e)


def get_env(env: str = "default", config_toml: dict[str, Any] | None = None) -> dict[str, Setting]:
    if config_toml is None:
        config_toml = app.load_config()

    env_section: dict[str, Any] = cast(dict[str, Any], config_toml.get("env", {}))
    env_data: dict[str, Any] = cast(dict[str, Any], env_section.get(env, None)) or {}

    if not env_data and env != "default":
        _logger.warning("Warning: user environment %s not found.", env)
        env_data = {}

    assert env_data is not None and isinstance(env_data, dict)

    return {
        "api_url": Setting(value=env_data.get("api_url", None)),
        "api_token": Setting(value=env_data.get("api_token", None), encrypted=True),
        "api_model_name": Setting(value=env_data.get("api_model_name", None)),
    }


def update_env(key: str, value: str | None, env: str = "default", config_toml: dict[str, Any] | None = None) -> dict[str, Any]:
    if config_toml is None:
        config_toml = app.load_config()

    envs_section: dict[str, Any] = cast(dict[str, Any], config_toml.setdefault("env", {}))
    env_config: dict[str, str] = cast(dict[str, str], envs_section.setdefault(env, {}))

    if key in ENCRYPTED_KEYS:
        value = encrypt_setting(value) if value is not None else None

    if value is None:
        env_config.pop(key, None)
    else:
        env_config[key] = value

    return env_config


def delete_env(env: str = "default", config_toml: dict[str, Any] | None = None):
    if config_toml is None:
        config_toml = app.load_config()

    envs_section: dict[str, Any] = cast(dict[str, Any], config_toml.setdefault("env", {}))
    envs_section.pop(env, None)

    _save_config_raw(config_toml)


def save_env(env_config: dict[str, str], env: str = "default", config_toml: dict[str, Any] | None = None):
    if config_toml is None:
        config_toml = app.load_config()

    envs_section: dict[str, Any] = cast(dict[str, Any], config_toml.setdefault("env", {}))
    envs_section[env] = env_config

    _save_config_raw(config_toml)
