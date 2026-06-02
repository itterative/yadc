"""Environment CRUD endpoints — backed by the ``cmd.envs`` module."""

from flask import jsonify, request

# cmd.envs is a heavy import (keyring, cryptography) — keep it at module level
# so it's loaded once, not on every request.
from yadc.cmd import envs as cmd_envs

from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import ApiBlueprint


@controller
def api_envs(app: ApiBlueprint, logging: LoggingFactory):
    _logger = logging.get_logger(__name__)

    @app.get("/envs")
    def list_envs():  # pyright: ignore[reportUnusedFunction]
        """List all environment names."""
        names = cmd_envs.list_all_env()
        return jsonify(names)

    @app.get("/envs/<name>")
    def get_env(name: str):  # pyright: ignore[reportUnusedFunction]
        """Return environment settings (token masked)."""
        settings = cmd_envs.get_env(name)

        api_url = settings.get("api_url")
        api_token = settings.get("api_token")
        api_model_name = settings.get("api_model_name")

        return jsonify(
            {
                "name": name,
                "api_url": api_url.value if api_url else None,
                "api_token": str(api_token) if api_token else None,  # Setting.__str__ masks encrypted values
                "api_model_name": api_model_name.value if api_model_name else None,
            }
        )

    @app.put("/envs/<name>")
    def put_env(name: str):  # pyright: ignore[reportUnusedFunction]
        """Create or update an environment.

        JSON body (all fields optional):
            api_url: str
            api_token: str
            api_model_name: str
        """
        body = request.get_json(silent=True) or {}

        # Build env config dict — only include keys that are present
        env_config: dict[str, str] = {}
        for key in cmd_envs.ENV_KEYS:
            if key in body:
                env_config[key] = body[key]

        cmd_envs.save_env(env_config, env=name)
        _logger.info("Environment '%s' saved.", name)

        return get_env(name)

    @app.delete("/envs/<name>")
    def delete_env(name: str):  # pyright: ignore[reportUnusedFunction]
        """Delete an environment."""
        if name == "default":
            return jsonify({"error": "Cannot delete the default environment"}), 400

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

        settings = cmd_envs.get_env(name)

        api_url = settings.get("api_url")
        if not api_url or not api_url.value:
            return jsonify({"error": "Environment has no API URL configured"}), 400

        api_token = settings.get("api_token")
        api_model_name = settings.get("api_model_name")

        url = f"{api_url.value.rstrip('/')}/models"
        headers: dict[str, str] = {}
        if api_token and api_token.value:
            headers["Authorization"] = f"Bearer {api_token.value}"

        try:
            resp = http_requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
        except http_requests.ConnectionError:
            return jsonify({"error": f"Could not connect to {url}"}), 502
        except http_requests.Timeout:
            return jsonify({"error": f"Connection to {url} timed out"}), 504
        except http_requests.HTTPError as e:
            return jsonify({"error": f"API returned {e.response.status_code}"}), 502
        except Exception as e:
            _logger.warning("Failed to fetch models from '%s': %s", url, e)
            return jsonify({"error": str(e)}), 502

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
            return jsonify({"error": "Could not parse model list from API response"}), 502

        models.sort()

        response_payload: dict[str, object] = {"models": models}
        if api_model_name and api_model_name.value:
            response_payload["default"] = api_model_name.value

        return jsonify(response_payload)
