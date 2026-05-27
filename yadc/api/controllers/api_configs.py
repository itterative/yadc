"""Dataset config CRUD endpoints — view and edit the TOML configs stored in STATE_PATH."""

import toml
from flask import jsonify, request
from pydantic import ValidationError

from yadc.api.services.datasets import DatasetService
from yadc.core.config import parse_config
from yadc.utils import deep_merge

from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import ApiBlueprint
from .models_errors import APIErrorDetail
from .utils_json import ErrorCode, jsonify_error


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
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        try:
            with open(info.config_path) as f:
                content = f.read()
        except FileNotFoundError:
            return jsonify_error(f"Config file not found for dataset '{name}'", status=404, code=ErrorCode.NOT_FOUND)
        except PermissionError:
            return jsonify_error("Permission denied", status=403, code=ErrorCode.PERMISSION_DENIED)

        # Parse to get structured fields alongside the raw text
        try:
            parsed = toml.loads(content)
        except Exception:
            parsed = {}

        validation_error = None
        if parsed:
            try:
                parse_config(parsed, strict=False)
            except ValidationError as e:
                validation_error = [{"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in e.errors()]

        return jsonify(
            {
                "name": name,
                "config_path": info.config_path,
                "content": content,
                "parsed": parsed,
                "validation_error": validation_error,
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
            return jsonify_error("Request body must include 'content'", status=400, code=ErrorCode.BAD_REQUEST)

        content = body["content"]
        if not isinstance(content, str):
            return jsonify_error("'content' must be a string", status=400, code=ErrorCode.BAD_REQUEST)

        # Validate it parses as TOML
        try:
            toml.loads(content)
        except Exception as e:
            return jsonify_error(f"Invalid TOML: {e}", status=400, code=ErrorCode.BAD_REQUEST)

        info = datasets.get_dataset(name)
        if info is None or info.config_path is None:
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        try:
            with open(info.config_path, "w") as f:
                f.write(content)
        except PermissionError:
            return jsonify_error("Permission denied", status=403, code=ErrorCode.PERMISSION_DENIED)

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
            return jsonify_error("Request body must be a JSON object", status=400, code=ErrorCode.BAD_REQUEST)

        info = datasets.get_dataset(name)
        if info is None or info.config_path is None:
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        # Read current config
        try:
            with open(info.config_path) as f:
                content = f.read()
        except FileNotFoundError:
            return jsonify_error(f"Config file not found for dataset '{name}'", status=404, code=ErrorCode.NOT_FOUND)
        except PermissionError:
            return jsonify_error("Permission denied", status=403, code=ErrorCode.PERMISSION_DENIED)

        try:
            parsed = toml.loads(content)
        except Exception as e:
            return jsonify_error(f"Existing config is invalid TOML: {e}", status=500, code=ErrorCode.INTERNAL_ERROR)

        # Deep-merge patch into existing config
        merged = deep_merge(parsed, body)

        # Validate merged result against the Config schema
        try:
            parse_config(merged, strict=False)
        except ValidationError as e:
            details = [APIErrorDetail.from_pydantic_error(err) for err in e.errors()]
            return jsonify_error("Validation failed", details=details, status=400, code=ErrorCode.VALIDATION_ERROR)

        # Serialize and validate round-trip
        try:
            new_content = toml.dumps(merged)
            toml.loads(new_content)  # round-trip validation
        except Exception as e:
            return jsonify_error(f"Failed to serialize config: {e}", status=400, code=ErrorCode.BAD_REQUEST)

        if dry_run:
            return jsonify({"name": name, "config_path": info.config_path, "content": new_content, "parsed": merged})

        try:
            with open(info.config_path, "w") as f:
                f.write(new_content)
        except PermissionError:
            return jsonify_error("Permission denied", status=403, code=ErrorCode.PERMISSION_DENIED)

        _logger.info("Config for dataset '%s' patched (fields: %s).", name, ", ".join(body.keys()))

        # Rescan so the index picks up any changes
        datasets.rescan_dataset(name)

        return jsonify({"name": name, "config_path": info.config_path, "content": new_content, "parsed": merged})

    @app.delete("/configs/<name>")
    def delete_config(name: str):  # pyright: ignore[reportUnusedFunction]
        """Delete a dataset config and unregister the dataset."""
        if not datasets.unregister_dataset(name):
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        _logger.info("Config for dataset '%s' deleted.", name)
        return jsonify({"status": "ok"})
