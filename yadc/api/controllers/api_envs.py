"""Environment CRUD endpoints — backed by the ``cmd.envs`` module."""

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
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_error, validate_body


class RevealEnvValueBody(pydantic.BaseModel):
    key: str
    password: str | None = None


class PutEnvBody(pydantic.BaseModel):
    api_url: str | None = None
    api_token: str | None = None
    api_model_name: str | None = None

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")


class ListModelsBody(pydantic.BaseModel):
    password: str | None = None


class PutKeyModeBody(pydantic.BaseModel):
    mode: Literal["keyring", "password"]
    password: str | None = None
    old_password: str | None = None


def _format_env(name: str, env_data: cmd_config.AppConfigEnv | None) -> dict[str, object]:
    """Format environment data for the API (token masked)."""
    api_url = env_data.api_url.value if env_data else None
    api_token = env_data.api_token if env_data else None
    api_model_name = env_data.api_model_name.value if env_data else None

    has_token = api_token is not None and api_token.value is not None

    return {
        "name": name,
        "api_url": api_url,
        "api_token": "[REDACTED]" if has_token else None,
        "api_model_name": api_model_name,
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
            password: str | null  (required when the value is password-encrypted)
        """
        body = validate_body(RevealEnvValueBody, await request.get_json(silent=True))

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
                decrypted = cmd_envs.decrypt_setting(value_obj.value, method=method, password=body.password)
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
            api_url: str
            api_token: str
            api_model_name: str
        """
        body = validate_body(PutEnvBody, await request.get_json(silent=True))
        config = cmd_config.load_config()

        for key, value in body.model_dump(exclude_none=True).items():
            cmd_envs.update_env(key, value, env=name, config=config)

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

    @app.route("/envs/<name>/models", methods=["GET", "POST"])
    async def list_models(name: str):  # pyright: ignore[reportUnusedFunction]
        """Fetch available models from the environment's API.

        Accepts both ``GET`` and ``POST``:

        - ``GET``: no body. The ``YADC_PASSWORD`` env var is the only way
          to decrypt a password-mode env's token.
        - ``POST``: optional ``{"password": "..."}`` body. Lets the client
          supply the decryption password explicitly when the backend can't
          read it from the env. Matches the body shape of
          ``POST /envs/<name>/reveal`` and ``PUT /envs/key-mode``.

        Both methods delegate to :func:`yadc.cmd.envs.models.list_models`,
        which decrypts the env's token and forwards the request to the
        captioner system's ``list_models`` helper. The captioner handles
        backend detection, HTTP retries, response parsing, and per-backend
        quirks (OpenAI / Gemini / Ollama / Koboldcpp / etc.).

        The cache TTL is read from ``Configuration.api_models_cache_ttl``.

        .. note::
            This dual-method route is intentionally awkward — it exists
            only because ``GET`` can't carry a body. The medium-term plan
            is to standardize on a single ``X-YADC-Password`` header
            across all password-passing endpoints. See the ``todo`` memory
            for details.
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

        # POST can carry an explicit password; GET can't.
        password: str | None = None
        raw_body = await request.get_json(silent=True)
        if request.method == "POST" and raw_body is not None:
            body = validate_body(ListModelsBody, raw_body)
            password = body.password

        try:
            models = await cmd_envs.list_models(
                name,
                password=password,
                cache_ttl=configuration.api_models_cache_ttl,
            )
        except PasswordRequiredError:
            return jsonify_error(
                "Password required to decrypt environment settings",
                status=403,
                code=ErrorCode.PASSWORD_REQUIRED,
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
            old_password: str | null  (current password when changing FROM password mode)
        """
        body = validate_body(PutKeyModeBody, await request.get_json(silent=True))

        try:
            config = cmd_config.load_config()
            cmd_envs.set_key_mode(config, body.mode, password=body.password, old_password=body.old_password)
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
