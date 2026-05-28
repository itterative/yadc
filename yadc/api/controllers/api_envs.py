"""Environment CRUD endpoints — backed by the ``cmd.envs`` module."""

from quart import jsonify, request

# cmd.envs is a heavy import (keyring, cryptography) — keep it at module level
# so it's loaded once, not on every request.
from yadc.cmd import config as cmd_config
from yadc.cmd import envs as cmd_envs
from yadc.core.env import YADC_PASSWORD

from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_error


def _format_env(name: str, env_data) -> dict[str, object]:
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
def api_envs(app: ApiBlueprint, logging: LoggingFactory):
    _logger = logging.get_logger(__name__)

    @app.get("/envs")
    def list_envs():  # pyright: ignore[reportUnusedFunction]
        """List all environments with full details."""
        names = cmd_envs.list_all_env()
        return jsonify([_format_env(name, cmd_envs.get_env(name)) for name in names])

    @app.get("/envs/<name>")
    def get_env(name: str):  # pyright: ignore[reportUnusedFunction]
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
        body = await request.get_json(silent=True) or {}
        key = body.get("key")
        password: str | None = body.get("password")

        if not key or key not in cmd_config.AppConfigEnv.model_fields:
            return jsonify_error("Invalid key", status=400, code=ErrorCode.BAD_REQUEST)

        env_data = cmd_envs.get_env(name)
        if env_data is None:
            return jsonify_error("Environment not found", status=404, code=ErrorCode.NOT_FOUND)

        value_obj = getattr(env_data, key)
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
            api_url: str
            api_token: str
            api_model_name: str
        """
        body = await request.get_json(silent=True) or {}

        config = cmd_config.load_config()

        for key in cmd_config.AppConfigEnv.model_fields:
            if key in body:
                cmd_envs.update_env(key, body[key], env=name, config=config)

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

    @app.post("/envs/<name>/models")
    def list_models(name: str):  # pyright: ignore[reportUnusedFunction]
        """Fetch available models from the environment's API.

        Proxies a ``GET /models`` request to the env's ``api_url`` so the
        frontend doesn't need direct CORS access to the inference API.
        """
        import requests as http_requests

        env_data = cmd_envs.get_env(name)
        if env_data is None:
            return jsonify_error("Environment not found", status=404, code=ErrorCode.NOT_FOUND)

        api_url = env_data.api_url.value
        if not api_url:
            return jsonify_error("Environment has no API URL configured", status=400, code=ErrorCode.BAD_REQUEST)

        api_token = env_data.api_token
        api_model_name = env_data.api_model_name.value

        url = f"{api_url.rstrip('/')}/models"
        headers: dict[str, str] = {}
        if api_token.value:
            if api_token.is_encrypted:
                method = cmd_envs.EncryptionMethod(api_token.method)
                try:
                    decrypted = cmd_envs.decrypt_setting(api_token.value, method=method)
                except cmd_envs.PasswordRequiredError:
                    return jsonify_error(
                        "Password required to decrypt environment settings",
                        status=403,
                        code=ErrorCode.PASSWORD_REQUIRED,
                    )
                if decrypted:
                    headers["Authorization"] = f"Bearer {decrypted}"
            else:
                headers["Authorization"] = f"Bearer {api_token.value}"

        try:
            resp = http_requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
        except http_requests.ConnectionError:
            return jsonify_error(f"Could not connect to {url}", status=502, code=ErrorCode.UPSTREAM_ERROR)
        except http_requests.Timeout:
            return jsonify_error(f"Connection to {url} timed out", status=504, code=ErrorCode.UPSTREAM_ERROR)
        except http_requests.HTTPError as e:
            return jsonify_error(f"API returned {e.response.status_code}", status=502, code=ErrorCode.UPSTREAM_ERROR)
        except Exception as e:
            _logger.warning("Failed to fetch models from '%s': %s", url, e)
            return jsonify_error(str(e), status=502, code=ErrorCode.UPSTREAM_ERROR)

        data = resp.json()

        # Normalize: extract model IDs from common response shapes
        models: list[str] = []

        # OpenAI-compatible: {"data": [{"id": "model-name", ...}, ...]}
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if isinstance(item, dict) and "id" in item:
                    models.append(item["id"])
        # Ollama: {"models": [{"name": "model-name", ...}, ...]}
        elif isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
            for item in data["models"]:
                if isinstance(item, dict) and "name" in item:
                    models.append(item["name"])
        # Fallback: plain list of strings
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    models.append(item)
                elif isinstance(item, dict):
                    # Try common keys
                    models.append(item.get("id") or item.get("name") or item.get("model", ""))
            models = [m for m in models if m]

        if not models:
            _logger.warning("Could not parse models from response: %s", type(data).__name__)
            return jsonify_error("Could not parse model list from API response", status=502, code=ErrorCode.UPSTREAM_ERROR)

        models.sort()

        response_payload: dict[str, object] = {"models": models}
        if api_model_name:
            response_payload["default"] = api_model_name

        return jsonify(response_payload)

    @app.get("/envs/key-mode")
    def get_key_mode():  # pyright: ignore[reportUnusedFunction]
        """Return the current key storage mode."""
        config = cmd_config.load_config()
        return jsonify({
            "mode": config.key_storage.mode,
            "env_password_set": bool(YADC_PASSWORD),
        })

    @app.put("/envs/key-mode")
    async def put_key_mode():  # pyright: ignore[reportUnusedFunction]
        """Switch the key storage mode or change the password for password mode.

        JSON body:
            mode: "keyring" | "password"
            password: str | null  (new password when mode="password")
            old_password: str | null  (current password when changing FROM password mode)
        """
        body = await request.get_json(silent=True) or {}
        mode = body.get("mode")

        if mode not in ("keyring", "password"):
            return jsonify_error("mode must be 'keyring' or 'password'", status=400, code=ErrorCode.BAD_REQUEST)

        password: str | None = body.get("password")
        old_password: str | None = body.get("old_password")

        try:
            config = cmd_config.load_config()
            cmd_envs.set_key_mode(config, mode, password=password, old_password=old_password)
            if config.key_storage.mode == mode and password is not None:
                _logger.info("Password changed for key storage mode '%s'.", mode)
            else:
                _logger.info("Key storage mode switched to '%s'.", mode)
            return jsonify({"mode": mode})
        except Exception as e:
            _logger.error("Failed to set key storage mode: %s", e)
            return jsonify_error(str(e), status=500, code=ErrorCode.INTERNAL_ERROR)
