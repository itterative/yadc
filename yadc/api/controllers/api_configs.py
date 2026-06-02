"""Dataset config CRUD endpoints — view and edit the TOML configs stored in STATE_PATH."""

import toml
from flask import jsonify, request

from yadc.api.services.datasets import DatasetService

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
    def get_config(name: str):  # pyright: ignore[reportUnusedFunction]
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

    @app.delete("/configs/<name>")
    def delete_config(name: str):  # pyright: ignore[reportUnusedFunction]
        """Delete a dataset config and unregister the dataset."""
        if not datasets.unregister_dataset(name):
            return jsonify({"error": f"Dataset '{name}' not found"}), 404

        _logger.info("Config for dataset '%s' deleted.", name)
        return jsonify({"status": "ok"})
