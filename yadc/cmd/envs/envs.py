"""Env CRUD using ``AppConfig`` (``list_all_env``, ``get_env``, ``load_env``, ``save_env``, ``update_env``, ``delete_env``, ``reset_envs``).

Each env is a named entry in ``AppConfig.envs`` (``api_url``,
``api_token``, ``api_model_name``, optional ``max_concurrent``). String
fields are wrapped in ``AppConfigEnvValue`` so each one carries its own
``method`` (none / keyring / password) and is encrypted / decrypted
independently — that's what lets a user rotate one env's key without
touching the others.

``load_env`` returns a fully-decrypted ``UserConfig`` for the named
env, falling back to the ``default`` env for any field that's missing
on the target. ``save_env`` / ``update_env`` accept raw string values
and encrypt them according to the current ``KeyStorage`` mode before
writing. ``reset_envs`` wipes every env (and is the only path that
doesn't ask for confirmation, used by ``yadc envs reset``).

``PasswordRequiredError`` is raised when an env value is encrypted with
the password mode but no password is available (neither in the
``password`` kwarg nor in ``YADC_PASSWORD``). The CLI surfaces it as a
prompt, the web UI as a 403 ``PASSWORD_REQUIRED`` error.
"""

import pydantic

from yadc.cmd.config import AppConfig, AppConfigEnv, AppConfigEnvValue, load_config, save_config
from yadc.core import logging
from yadc.core.user_config import UserConfig, UserConfigApi

from .encryption import _active_storage_method, decrypt_setting, encrypt_setting
from .keystorage_password import PasswordRequiredError
from .setting import EncryptionMethod


class EnvDecryptionError(ValueError):
    """Raised when an environment setting cannot be decrypted."""


_logger = logging.get_logger(__name__)


def list_all_env() -> list[str]:
    config = load_config()
    return list(config.envs.keys())


def reset_envs() -> None:
    config = load_config()
    config.envs = {}
    save_config(config)


def _decrypt_env_value(key: str, value_obj: AppConfigEnvValue, env_name: str, password: str | None = None) -> str | None:
    """Decrypt a single env value if it is encrypted."""
    if value_obj.value is None:
        return None
    if value_obj.method == "none":
        return value_obj.value

    method = EncryptionMethod(value_obj.method)
    decrypted = decrypt_setting(value_obj.value, method=method, password=password)
    if decrypted is None:
        raise EnvDecryptionError(
            f"Failed to decrypt '{key}' for environment '{env_name}'. Ensure your keyring is unlocked (e.g., gpg-agent is running with a working pinentry)."
        )
    return decrypted


def load_env(env: str = "default", password: str | None = None) -> UserConfig:
    try:
        config = load_config()
        default_env = get_env("default", config=config) or AppConfigEnv()
        target_env = get_env(env, config=config) or AppConfigEnv()

        def _get_value(field: AppConfigEnvValue, fallback: AppConfigEnvValue, key: str) -> str | None:
            value_obj = field if field.value is not None else fallback
            if value_obj.value is None:
                return None
            return _decrypt_env_value(key, value_obj, env, password=password)

        api_url = _get_value(target_env.api_url, default_env.api_url, "api_url")
        api_token = _get_value(target_env.api_token, default_env.api_token, "api_token")
        api_model_name = _get_value(target_env.api_model_name, default_env.api_model_name, "api_model_name")
        # ``max_concurrent`` falls back from the target env to the
        # default env (same pattern as the other env fields). Plain
        # int — no decryption.
        max_concurrent = target_env.max_concurrent if target_env.max_concurrent is not None else default_env.max_concurrent

        return UserConfig(
            api=UserConfigApi(
                url=api_url or "",
                token=api_token or "",
                model_name=api_model_name or "",
                max_concurrent=max_concurrent,
            ),
        )
    except EnvDecryptionError:
        raise
    except PasswordRequiredError:
        raise
    except (pydantic.ValidationError, ValueError):
        _logger.warning("Warning: user config is invalid")
    except PermissionError:
        _logger.warning("Warning: failed to read user config: permission denied")
    except Exception as e:
        _logger.warning("Warning: failed to read user config: %s", e)

    return UserConfig(api=UserConfigApi())


def get_env(env: str = "default", config: AppConfig | None = None) -> AppConfigEnv | None:
    if config is None:
        config = load_config()

    env_data = config.envs.get(env)
    if env_data is None and env != "default":
        _logger.warning("Warning: user environment %s not found.", env)
    return env_data


def update_env(
    key: str,
    value: str | None,
    env: str = "default",
    config: AppConfig | None = None,
) -> None:
    if config is None:
        config = load_config()

    env_data = config.envs.get(env)
    if env_data is None:
        env_data = AppConfigEnv()
        config.envs[env] = env_data

    current = getattr(env_data, key)
    if isinstance(current, AppConfigEnvValue):
        # String-valued field (api_url, api_token, api_model_name):
        # route through the encryption layer. ``None`` clears the
        # value and resets the encryption method to "none".
        if value is None:
            current.value = None
            current.method = "none"
        elif key in AppConfigEnv.ENCRYPTED_FIELDS:
            method = _active_storage_method()
            current.value = encrypt_setting(value, method=method)
            current.method = method.value
        else:
            current.value = value
            current.method = "none"
        return

    # Non-string field (e.g. ``max_concurrent: int | None``): plain
    # assignment. No encryption, no validation beyond what the model
    # performs on construction.
    if key not in AppConfigEnv.model_fields:
        raise ValueError(f"Invalid environment key: {key}")
    setattr(env_data, key, value)


def delete_env(env: str = "default", config: AppConfig | None = None) -> None:
    if config is None:
        config = load_config()

    config.envs.pop(env, None)
    save_config(config)


def save_env(config: AppConfig | None = None) -> None:
    if config is None:
        config = load_config()
    save_config(config)
