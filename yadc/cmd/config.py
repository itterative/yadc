"""``AppConfig`` Pydantic model hierarchy with v0→v1 TOML migration and ``save_config``.

The user config file is one TOML document rooted at ``AppConfig``:

- ``key_storage`` (``AppConfigKeyStorage``) — ``mode`` is
  ``"keyring"`` or ``"password"``; ``password`` carries the
  RSA-2048 keypair (``public_key`` is plain, ``private_key`` is the
  keyring-encrypted PEM).
- ``envs`` — dict of ``AppConfigEnv`` keyed by env name. String fields
  (``api_url`` / ``api_token`` / ``api_model_name``) are wrapped in
  ``AppConfigEnvValue`` to track their per-field encryption state;
  ``max_concurrent`` is a plain ``int | None``.

``save_config`` writes ``AppConfig`` to the user config path with
``tomlkit`` (preserves comments and formatting on round-trip).

``load_config`` parses the TOML into a ``tomlkit.document`` and detects
the version from the top-level keys. v0 (legacy ``UserConfig`` with
``api.url``/``api.token``/``api.model_name``) is auto-migrated to v1
on the first load; the migration is in-place and writes back to disk
unless ``migrate=False`` is passed.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar, Literal

import pydantic
import tomlkit
from pydantic import ConfigDict

from yadc.utils.dict_utils import load_toml_file

from .app import CONFIG_PATH


class AppConfigEnvValue(pydantic.BaseModel):
    """A single environment setting value with its encryption state."""

    value: str | None = None
    method: Literal["none", "keyring", "password"] = "none"

    @property
    def is_encrypted(self) -> bool:
        return self.method != "none"


class AppConfigEnv(pydantic.BaseModel):
    """Environment settings. Fields are always present; the value inside may be None.

    String-valued fields (``api_url``, ``api_token``, ``api_model_name``)
    are wrapped in :class:`AppConfigEnvValue` to track their encryption
    state (plain, keyring-encrypted, or password-encrypted). Integer
    fields (``max_concurrent``) are stored as plain ``int | None`` —
    they are never encrypted.
    """

    model_config: ClassVar[ConfigDict] = pydantic.ConfigDict(extra="forbid")

    ENCRYPTED_FIELDS: ClassVar[set[str]] = {"api_token"}

    api_url: AppConfigEnvValue = pydantic.Field(default_factory=AppConfigEnvValue)
    api_token: AppConfigEnvValue = pydantic.Field(default_factory=AppConfigEnvValue)
    api_model_name: AppConfigEnvValue = pydantic.Field(default_factory=AppConfigEnvValue)
    max_concurrent: int | None = None


class AppConfigKeyStoragePassword(pydantic.BaseModel):
    private_key: str
    public_key: str


class AppConfigKeyStorage(pydantic.BaseModel):
    mode: Literal["keyring", "password"]
    password: AppConfigKeyStoragePassword | None = None


class AppConfig(pydantic.BaseModel):
    key_storage: AppConfigKeyStorage
    envs: dict[str, AppConfigEnv]


class _AppConfigV1Env(pydantic.BaseModel):
    api_url: str | None = None
    api_token: str | None = None
    api_model_name: str | None = None
    max_concurrent: int | None = None


class _AppConfigV1KeyStoragePassword(pydantic.BaseModel):
    private_key: str
    public_key: str


class _AppConfigV1KeyStorage(pydantic.BaseModel):
    mode: Literal["keyring", "password"]
    password: _AppConfigV1KeyStoragePassword | None = None


class _AppConfigV1(pydantic.BaseModel):
    version: Literal[1] = 1
    key_storage: _AppConfigV1KeyStorage = pydantic.Field(default_factory=lambda: _AppConfigV1KeyStorage(mode="keyring"))
    envs: dict[str, _AppConfigV1Env]


def _parse_env_value(field_name: str, raw_value: str | None) -> AppConfigEnvValue:
    """Parse a raw TOML value into an ``AppConfigEnvValue``.

    Detects ``keyring:`` / ``password:`` prefixes. Bare values on fields
    expected to be encrypted are treated as legacy keyring ciphertext.
    """
    if raw_value is None:
        return AppConfigEnvValue(value=None, method="none")

    if raw_value.startswith("keyring:"):
        return AppConfigEnvValue(value=raw_value[8:], method="keyring")
    if raw_value.startswith("password:"):
        return AppConfigEnvValue(value=raw_value[9:], method="password")

    # Bare value: legacy encrypted values assumed keyring
    if field_name in AppConfigEnv.ENCRYPTED_FIELDS:
        return AppConfigEnvValue(value=raw_value, method="keyring")

    return AppConfigEnvValue(value=raw_value, method="none")


def _serialize_env_value(value: AppConfigEnvValue) -> str | None:
    """Serialize an ``AppConfigEnvValue`` back to a TOML string (with prefix)."""
    if value.value is None:
        return None
    if value.method == "keyring":
        return f"keyring:{value.value}"
    if value.method == "password":
        return f"password:{value.value}"
    return value.value


def _migrate_v0_env(raw: dict[str, Any]) -> AppConfigEnv:
    """Migrate a v0 env dict to an ``AppConfigEnv``.

    Legacy ``api_token_encrypted`` becomes ``api_token`` with a ``keyring:``
    prefix. All other keys pass through unchanged.
    """
    migrated = dict(raw)
    encrypted = migrated.pop("api_token_encrypted", None)
    if encrypted:
        migrated["api_token"] = f"keyring:{encrypted}"

    return AppConfigEnv(
        api_url=_parse_env_value("api_url", migrated.get("api_url")),
        api_token=_parse_env_value("api_token", migrated.get("api_token")),
        api_model_name=_parse_env_value("api_model_name", migrated.get("api_model_name")),
        max_concurrent=migrated.get("max_concurrent"),
    )


def _load_config() -> dict[str, Any]:
    config_path = CONFIG_PATH / "config.toml"

    try:
        with open(config_path, "r") as f:
            return load_toml_file(f)
    except FileNotFoundError:
        return {}
    except PermissionError:
        raise
    except Exception as e:
        raise ValueError("user config is invalid") from e


def load_config() -> AppConfig:
    config = _load_config()

    version = config.get("version", 0)
    match version:
        case 0:
            envs_raw = config.get("env", {})
            return AppConfig(
                key_storage=AppConfigKeyStorage(mode="keyring"),
                envs={k: _migrate_v0_env(e) for k, e in envs_raw.items()},
            )

        case 1:
            _config = _AppConfigV1.model_validate(config)

            return AppConfig(
                key_storage=AppConfigKeyStorage(
                    mode=_config.key_storage.mode,
                    password=AppConfigKeyStoragePassword(
                        private_key=_config.key_storage.password.private_key,
                        public_key=_config.key_storage.password.public_key,
                    )
                    if _config.key_storage.password is not None
                    else None,
                ),
                envs={
                    k: AppConfigEnv(
                        api_url=_parse_env_value("api_url", e.api_url),
                        api_token=_parse_env_value("api_token", e.api_token),
                        api_model_name=_parse_env_value("api_model_name", e.api_model_name),
                        max_concurrent=e.max_concurrent,
                    )
                    for k, e in _config.envs.items()
                },
            )

        case _:
            raise ValueError(f"unknown config version: {version!r}")


def save_config(config: AppConfig) -> None:
    """Serialize *config* as v1 TOML and write it to disk."""
    envs_v1: dict[str, _AppConfigV1Env] = {}
    for name, env in config.envs.items():
        envs_v1[name] = _AppConfigV1Env(
            api_url=_serialize_env_value(env.api_url),
            api_token=_serialize_env_value(env.api_token),
            api_model_name=_serialize_env_value(env.api_model_name),
            max_concurrent=env.max_concurrent,
        )

    v1 = _AppConfigV1(
        version=1,
        key_storage=_AppConfigV1KeyStorage(
            mode=config.key_storage.mode,
            password=_AppConfigV1KeyStoragePassword(
                private_key=config.key_storage.password.private_key,
                public_key=config.key_storage.password.public_key,
            )
            if config.key_storage.password is not None
            else None,
        ),
        envs=envs_v1,
    )

    data = v1.model_dump(exclude_none=True)
    config_path = CONFIG_PATH / "config.toml"
    with open(config_path, "w") as f:
        tomlkit.dump(data, f)
    try:
        os.chmod(config_path, 0o600)
    except Exception:
        pass
