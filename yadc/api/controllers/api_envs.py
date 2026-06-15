"""Environment CRUD endpoints — backed by the ``cmd.envs`` module."""

import asyncio
from typing import ClassVar, Literal

import httpx
import pydantic
from quart import jsonify, request

from yadc.cmd import config as cmd_config
from yadc.cmd import envs as cmd_envs
from yadc.cmd.envs.keystorage_password import PasswordRequiredError
from yadc.core.env import YADC_PASSWORD

from ..configuration import Configuration
from ..modules.logging_factory import LoggingFactory
from . import controller
from ._password import resolve_request_password
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_error, validate_body


class RevealEnvValueBody(pydantic.BaseModel):
    key: str


class PutEnvBody(pydantic.BaseModel):
    api_url: str | None = None
    api_token: str | None = None
    api_model_name: str | None = None
    max_concurrent: int | None = pydantic.Field(default=None, ge=1)

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")


class PutKeyModeBody(pydantic.BaseModel):
    """Body for ``PUT /api/envs/key-mode``.

    ``password`` is the **new** password (a value being submitted). The
    **current** password comes from the ``yadc_password`` session
    cookie (with the ``YADC_PASSWORD`` env-var fallback) — see
    :func:`_password.resolve_request_password`.
    """

    mode: Literal["keyring", "password"]
    password: str | None = None


def _format_env(name: str, env_data: cmd_config.AppConfigEnv | None) -> dict[str, object]:
    """Format environment data for the API (token masked)."""
    api_url = env_data.api_url.value if env_data else None
    api_token = env_data.api_token if env_data else None
    api_model_name = env_data.api_model_name.value if env_data else None
    max_concurrent = env_data.max_concurrent if env_data else None

    has_token = api_token is not None and api_token.value is not None

    return {
        "name": name,
        "api_url": api_url,
        "api_token": "[REDACTED]" if has_token else None,
        "api_model_name": api_model_name,
        "max_concurrent": max_concurrent,
        "has_token": has_token,
        "token_method": api_token.method if api_token is not None else None,
    }


@controller
def api_envs(app: ApiBlueprint, configuration: Configuration, logging: LoggingFactory):
    _logger = logging.get_logger(__name__)

    @app.get("/envs")
    def list_envs():  # pyright: ignore[reportUnusedFunction]
        """List all environments with full details."""
        names = cmd_envs.list_all_env()
        return jsonify([_format_env(name, cmd_envs.get_env(name)) for name in names])

    @app.get("/envs/<name>")
    def get_env(name: str):
        """Return environment settings (token masked)."""
        env_data = cmd_envs.get_env(name)
        return jsonify(_format_env(name, env_data))

    @app.post("/envs/<name>/reveal")
    async def reveal_env_value(name: str):  # pyright: ignore[reportUnusedFunction]
        """Reveal the unredacted value of a specific environment key.

        JSON body:
            key: str

        The decryption password is read from the ``yadc_password`` session
        cookie (with the ``YADC_PASSWORD`` env-var fallback). Returns
        403 ``PASSWORD_REQUIRED`` if neither is set and the value is
        password-encrypted.
        """
        body = validate_body(RevealEnvValueBody, await request.get_json(silent=True))
        password = resolve_request_password(request)

        if body.key not in cmd_config.AppConfigEnv.model_fields:
            return jsonify_error("Invalid key", status=400, code=ErrorCode.BAD_REQUEST)

        env_data = cmd_envs.get_env(name)
        if env_data is None:
            return jsonify_error("Environment not found", status=404, code=ErrorCode.NOT_FOUND)

        value_obj = getattr(env_data, body.key)
        if not isinstance(value_obj, cmd_config.AppConfigEnvValue):
            return jsonify_error("Invalid key", status=400, code=ErrorCode.BAD_REQUEST)

        if value_obj.value is None:
            return jsonify_error("Key not set", status=404, code=ErrorCode.NOT_FOUND)

        if value_obj.is_encrypted:
            method = cmd_envs.EncryptionMethod(value_obj.method)
            try:
                decrypted = cmd_envs.decrypt_setting(value_obj.value, method=method, password=password)
            except cmd_envs.PasswordRequiredError:
                return jsonify_error(
                    "Password required to decrypt environment settings",
                    status=403,
                    code=ErrorCode.PASSWORD_REQUIRED,
                )
            if decrypted is None:
                return jsonify_error("Failed to decrypt value", status=500, code=ErrorCode.INTERNAL_ERROR)
            return jsonify({"value": decrypted})

        return jsonify({"value": value_obj.value})

    @app.put("/envs/<name>")
    async def put_env(name: str):  # pyright: ignore[reportUnusedFunction]
        """Create or update an environment.

        JSON body (all fields optional):
            api_url: str | null          -- ``null`` clears the URL
            api_token: str | null        -- ``null`` clears the token
            api_model_name: str | null   -- ``null`` clears the default model
            max_concurrent: int | null   -- ``null`` clears the concurrency default

        Fields omitted from the body are left untouched; fields explicitly
        set to ``null`` are cleared (mirroring ``yadc envs delete <key>``).
        Fields set to a string are stored as-is (token is re-encrypted
        transparently by ``cmd_envs.update_env``).
        """
        body = validate_body(PutEnvBody, await request.get_json(silent=True))
        config = cmd_config.load_config()

        for key in body.model_fields_set:
            cmd_envs.update_env(key, getattr(body, key), env=name, config=config)

        cmd_envs.save_env(config=config)
        _logger.info("Environment '%s' saved.", name)

        return get_env(name)

    @app.delete("/envs/<name>")
    def delete_env(name: str):  # pyright: ignore[reportUnusedFunction]
        """Delete an environment."""
        if name == "default":
            return jsonify_error("Cannot delete the default environment", status=400, code=ErrorCode.BAD_REQUEST)

        cmd_envs.delete_env(env=name)
        _logger.info("Environment '%s' deleted.", name)
        return jsonify({"status": "ok"})

    @app.get("/envs/<name>/models")
    async def list_models(name: str):  # pyright: ignore[reportUnusedFunction]
        """Fetch available models from the environment's API.

        The decryption password is read from the ``yadc_password`` session
        cookie (with the ``YADC_PASSWORD`` env-var fallback), so the
        ``GET`` carries everything it needs without a body.

        Delegates to :func:`yadc.cmd.envs.models.list_models`, which
        decrypts the env's token and forwards the request to the
        captioner system's ``list_models`` helper. The captioner handles
        backend detection, HTTP retries, response parsing, and
        per-backend quirks (OpenAI / Gemini / Ollama / Koboldcpp / etc.).

        The cache TTL is read from ``Configuration.api_models_cache_ttl``,
        and the overall operation is bounded by
        ``Configuration.list_models_timeout`` — exceeding it surfaces as
        HTTP 504 GATEWAY_TIMEOUT so a dead/slow env doesn't leave the
        model picker spinning forever.
        """
        env_data = cmd_envs.get_env(name)
        if env_data is None:
            return jsonify_error("Environment not found", status=404, code=ErrorCode.NOT_FOUND)

        # Distinguish a misconfigured env (no api_url) from an upstream
        # error — the former is a 400 (client fix), the latter is a 502.
        # The orchestrator's ValueError covers both, so we short-circuit
        # here to keep the status codes meaningful.
        if not env_data.api_url.value:
            return jsonify_error("Environment has no API URL configured", status=400, code=ErrorCode.BAD_REQUEST)

        password = resolve_request_password(request)

        try:
            models = await cmd_envs.list_models(
                name,
                password=password,
                cache_ttl=configuration.api_models_cache_ttl,
                timeout=configuration.list_models_timeout,
            )
        except PasswordRequiredError:
            return jsonify_error(
                "Password required to decrypt environment settings",
                status=403,
                code=ErrorCode.PASSWORD_REQUIRED,
            )
        except asyncio.TimeoutError:
            return jsonify_error(
                f"Timed out fetching model list after {configuration.list_models_timeout} seconds — check the API URL or network",
                status=504,
                code=ErrorCode.GATEWAY_TIMEOUT,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            return jsonify_error(f"Could not reach API: {e}", status=502, code=ErrorCode.UPSTREAM_ERROR)
        except httpx.HTTPStatusError as e:
            return jsonify_error(f"API returned {e.response.status_code}", status=502, code=ErrorCode.UPSTREAM_ERROR)
        except ValueError as e:
            _logger.warning("Failed to fetch models for env '%s': %s", name, e)
            return jsonify_error(str(e), status=502, code=ErrorCode.UPSTREAM_ERROR)

        response_payload: dict[str, object] = {"models": models}
        if env_data.api_model_name.value:
            response_payload["default"] = env_data.api_model_name.value

        return jsonify(response_payload)

    @app.get("/envs/key-mode")
    def get_key_mode():  # pyright: ignore[reportUnusedFunction]
        """Return the current key storage mode."""
        config = cmd_config.load_config()
        return jsonify(
            {
                "mode": config.key_storage.mode,
                "env_password_set": bool(YADC_PASSWORD),
            }
        )

    @app.put("/envs/key-mode")
    async def put_key_mode():  # pyright: ignore[reportUnusedFunction]
        """Switch the key storage mode or change the password for password mode.

        JSON body:
            mode: "keyring" | "password"
            password: str | null  (new password when mode="password")

        The **current** password is read from the ``yadc_password``
        session cookie (with the ``YADC_PASSWORD`` env-var fallback).
        It's needed when switching FROM password mode to keyring.
        """
        body = validate_body(PutKeyModeBody, await request.get_json(silent=True))
        old_password = resolve_request_password(request)

        try:
            config = cmd_config.load_config()
            cmd_envs.set_key_mode(config, body.mode, password=body.password, old_password=old_password)
            if config.key_storage.mode == body.mode and body.password is not None:
                _logger.info("Password changed for key storage mode '%s'.", body.mode)
            else:
                _logger.info("Key storage mode switched to '%s'.", body.mode)
            return jsonify({"mode": body.mode})
        except PasswordRequiredError as e:
            _logger.warning("Wrong password supplied for key mode change: %s", e)
            return jsonify_error(
                "Current password is incorrect.",
                status=403,
                code=ErrorCode.PASSWORD_REQUIRED,
            )
        except Exception as e:
            _logger.error("Failed to set key storage mode: %s", e)
            return jsonify_error(str(e), status=500, code=ErrorCode.INTERNAL_ERROR)
