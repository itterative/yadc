"""Dataset config CRUD endpoints — view and edit the TOML configs stored in STATE_PATH."""

import toml
from flask import jsonify, request

from yadc.api.services.datasets import DatasetService
from yadc.utils import deep_merge

from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import ApiBlueprint


@controller
def api_configs(app: ApiBlueprint, logging: LoggingFactory, datasets: DatasetService):
    _logger = logging.get_logger(__name__)

    @app.get("/configs")
    def list_configs():  # pyright: ignore[reportUnusedFunction]
        """List all dataset config names."""
        ds_list = datasets.list_datasets()
        return jsonify([{"name": d.name, "config_path": d.config_path} for d in ds_list])

    @app.get("/configs/<name>")
    def get_config(name: str):
        """Return the raw TOML content of a dataset config."""
        info = datasets.get_dataset(name)
        if info is None or info.config_path is None:
            return jsonify({"error": f"Dataset '{name}' not found"}), 404

        try:
            with open(info.config_path) as f:
                content = f.read()
        except FileNotFoundError:
            return jsonify({"error": f"Config file not found for dataset '{name}'"}), 404
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403

        # Parse to get structured fields alongside the raw text
        try:
            parsed = toml.loads(content)
        except Exception:
            parsed = {}

        return jsonify(
            {
                "name": name,
                "config_path": info.config_path,
                "content": content,
                "parsed": parsed,
            }
        )

    @app.put("/configs/<name>")
    def put_config(name: str):  # pyright: ignore[reportUnusedFunction]
        """Update a dataset config's raw TOML content.

        JSON body: {"content": "..."} (raw TOML string)
        The TOML is validated before saving.
        """
        body = request.get_json(silent=True)
        if body is None or "content" not in body:
            return jsonify({"error": "Request body must include 'content'"}), 400

        content = body["content"]
        if not isinstance(content, str):
            return jsonify({"error": "'content' must be a string"}), 400

        # Validate it parses as TOML
        try:
            toml.loads(content)
        except Exception as e:
            return jsonify({"error": f"Invalid TOML: {e}"}), 400

        info = datasets.get_dataset(name)
        if info is None or info.config_path is None:
            return jsonify({"error": f"Dataset '{name}' not found"}), 404

        try:
            with open(info.config_path, "w") as f:
                f.write(content)
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403

        _logger.info("Config for dataset '%s' updated.", name)

        # Rescan so the index picks up any path/data changes
        datasets.rescan_dataset(name)

        return get_config(name)

    @app.patch("/configs/<name>")
    def patch_config(name: str):  # pyright: ignore[reportUnusedFunction]
        """Partially update a dataset config using a JSON body.

        JSON body: a partial dict matching the TOML structure (e.g.
        `{"settings": {"max_tokens": 1024}, "rounds": 2}`).
        The patch is deep-merged into the existing parsed config, then
        serialized back to TOML.

        Query params:
            dry_run: If "1" or "true", return the merged result without
                     writing to disk or rescanning.
        """
        dry_run = request.args.get("dry_run", "").lower() in ("1", "true")

        body = request.get_json(silent=True)
        if body is None or not isinstance(body, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        info = datasets.get_dataset(name)
        if info is None or info.config_path is None:
            return jsonify({"error": f"Dataset '{name}' not found"}), 404

        # Read current config
        try:
            with open(info.config_path) as f:
                content = f.read()
        except FileNotFoundError:
            return jsonify({"error": f"Config file not found for dataset '{name}'"}), 404
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403

        try:
            parsed = toml.loads(content)
        except Exception as e:
            return jsonify({"error": f"Existing config is invalid TOML: {e}"}), 500

        # Deep-merge patch into existing config
        merged = deep_merge(parsed, body)

        # Serialize and validate
        try:
            new_content = toml.dumps(merged)
            toml.loads(new_content)  # round-trip validation
        except Exception as e:
            return jsonify({"error": f"Failed to serialize config: {e}"}), 400

        if dry_run:
            return jsonify({"name": name, "config_path": info.config_path, "content": new_content, "parsed": merged})

        try:
            with open(info.config_path, "w") as f:
                f.write(new_content)
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403

        _logger.info("Config for dataset '%s' patched (fields: %s).", name, ", ".join(body.keys()))

        # Rescan so the index picks up any changes
        datasets.rescan_dataset(name)

        return jsonify({"name": name, "config_path": info.config_path, "content": new_content, "parsed": merged})

    @app.delete("/configs/<name>")
    def delete_config(name: str):  # pyright: ignore[reportUnusedFunction]
        """Delete a dataset config and unregister the dataset."""
        if not datasets.unregister_dataset(name):
            return jsonify({"error": f"Dataset '{name}' not found"}), 404

        _logger.info("Config for dataset '%s' deleted.", name)
        return jsonify({"status": "ok"})
