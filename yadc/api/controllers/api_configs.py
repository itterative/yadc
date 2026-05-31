"""Dataset config CRUD endpoints — view and edit the TOML configs stored in STATE_PATH."""

from typing import Any

import tomlkit
from pydantic import ValidationError
from quart import jsonify, request

from yadc.api.services.config_history import ConfigHistoryService
from yadc.api.services.datasets import DatasetService
from yadc.core.config import parse_config
from yadc.utils.dict_utils import toml_merge, toml_to_plain

from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import ApiBlueprint
from .models_errors import APIErrorDetail
from .utils_json import ErrorCode, jsonify_error


def _extract_dataset_paths(doc: dict[str, Any]) -> set[str]:
    """Extract path values from [[dataset]] entries."""
    entries = doc.get("dataset")
    if not isinstance(entries, list):
        return set()
    return {str(entry.get("path", "")) for entry in entries if isinstance(entry, dict)}


@controller
def api_configs(
    app: ApiBlueprint,
    logging: LoggingFactory,
    datasets: DatasetService,
    config_history: ConfigHistoryService,
):
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
            parsed = tomlkit.loads(content)
        except Exception:
            parsed = {}

        validation_error = None
        if parsed:
            try:
                parse_config(toml_to_plain(parsed), strict=False)
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
    async def put_config(name: str):  # pyright: ignore[reportUnusedFunction]
        """Update a dataset config's raw TOML content.

        JSON body: {"content": "..."} (raw TOML string)
        The TOML is validated before saving.
        """
        body = await request.get_json(silent=True)
        if body is None or "content" not in body:
            return jsonify_error("Request body must include 'content'", status=400, code=ErrorCode.BAD_REQUEST)

        content = body["content"]
        if not isinstance(content, str):
            return jsonify_error("'content' must be a string", status=400, code=ErrorCode.BAD_REQUEST)

        # Validate it parses as TOML
        try:
            new_doc = tomlkit.loads(content)
        except Exception as e:
            return jsonify_error(f"Invalid TOML: {e}", status=400, code=ErrorCode.BAD_REQUEST)

        info = datasets.get_dataset(name)
        if info is None or info.config_path is None:
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        # Managed datasets: prevent path changes
        if info.source == "upload":
            try:
                with open(info.config_path) as f:
                    original_doc = tomlkit.loads(f.read())
            except Exception:
                original_doc = {}
            original_paths = _extract_dataset_paths(toml_to_plain(original_doc))
            new_paths = _extract_dataset_paths(toml_to_plain(new_doc))
            if original_paths != new_paths:
                return jsonify_error(
                    "Managed dataset paths cannot be changed. Use the upload panel to add or remove images.",
                    status=400,
                    code=ErrorCode.BAD_REQUEST,
                )

        try:
            with open(info.config_path, "w") as f:
                f.write(content)
        except PermissionError:
            return jsonify_error("Permission denied", status=403, code=ErrorCode.PERMISSION_DENIED)

        _logger.info("Config for dataset '%s' updated.", name)

        # Snapshot the new content so the current state is always recoverable
        config_history.save_snapshot(name, content)

        # Rescan so the index picks up any path/data changes
        datasets.rescan_dataset(name)

        return get_config(name)

    @app.patch("/configs/<name>")
    async def patch_config(name: str):  # pyright: ignore[reportUnusedFunction]
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

        body = await request.get_json(silent=True)
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
            parsed = tomlkit.loads(content)
        except Exception as e:
            return jsonify_error(f"Existing config is invalid TOML: {e}", status=500, code=ErrorCode.INTERNAL_ERROR)

        # Deep-merge patch into existing config (preserves comments/formatting)
        merged_doc = toml_merge(parsed, body)

        # Validate merged result against the Config schema
        # Use plain dict conversion — Pydantic rejects tomlkit wrapper types
        try:
            parse_config(toml_to_plain(merged_doc), strict=False)
        except ValidationError as e:
            details = [APIErrorDetail.from_pydantic_error(err) for err in e.errors()]
            return jsonify_error("Validation failed", details=details, status=400, code=ErrorCode.VALIDATION_ERROR)

        # Managed datasets: prevent path changes (check before writing, including dry_run)
        if info.source == "upload":
            original_paths = _extract_dataset_paths(toml_to_plain(parsed))
            new_paths = _extract_dataset_paths(toml_to_plain(merged_doc))
            if original_paths != new_paths:
                return jsonify_error(
                    "Managed dataset paths cannot be changed. Use the upload panel to add or remove images.",
                    status=400,
                    code=ErrorCode.BAD_REQUEST,
                )

        # Serialize and validate round-trip
        try:
            new_content = tomlkit.dumps(merged_doc)
            tomlkit.loads(new_content)  # round-trip validation
        except Exception as e:
            return jsonify_error(f"Failed to serialize config: {e}", status=400, code=ErrorCode.BAD_REQUEST)

        if dry_run:
            return jsonify({"name": name, "config_path": info.config_path, "content": new_content, "parsed": toml_to_plain(merged_doc)})

        try:
            with open(info.config_path, "w") as f:
                f.write(new_content)
        except PermissionError:
            return jsonify_error("Permission denied", status=403, code=ErrorCode.PERMISSION_DENIED)

        _logger.info("Config for dataset '%s' patched (fields: %s).", name, ", ".join(body.keys()))

        # Snapshot the new content so the current state is always recoverable
        config_history.save_snapshot(name, new_content)

        # Rescan so the index picks up any changes
        datasets.rescan_dataset(name)

        return jsonify({"name": name, "config_path": info.config_path, "content": new_content, "parsed": toml_to_plain(merged_doc)})

    @app.get("/configs/<name>/history")
    def list_config_history(name: str):  # pyright: ignore[reportUnusedFunction]
        """List config revision history for a dataset."""
        info = datasets.get_dataset(name)
        if info is None or info.config_path is None:
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        limit = request.args.get("limit", 50, type=int)
        before_id = request.args.get("before_id", None, type=int)

        entries = config_history.list_history(name, limit=limit, before_id=before_id)
        return jsonify(
            [
                {
                    "id": e.id,
                    "dataset_name": e.dataset_name,
                    "content": e.content,
                    "created_t": e.created_t,
                }
                for e in entries
            ]
        )

    @app.post("/configs/<name>/history/<int:entry_id>/restore")
    async def restore_config_history(name: str, entry_id: int):  # pyright: ignore[reportUnusedFunction]
        """Restore a config from a history snapshot.

        Saves the current config to history first, then overwrites with the
        historical version.
        """
        info = datasets.get_dataset(name)
        if info is None or info.config_path is None:
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        entry = config_history.get_entry(entry_id)
        if entry is None or entry.dataset_name != name:
            return jsonify_error(f"History entry {entry_id} not found for dataset '{name}'", status=404, code=ErrorCode.NOT_FOUND)

        # Save current content to history before restoring
        try:
            with open(info.config_path) as f:
                current_content = f.read()
            config_history.save_snapshot(name, current_content)
        except FileNotFoundError:
            pass

        # Write the historical content
        try:
            with open(info.config_path, "w") as f:
                f.write(entry.content)
        except PermissionError:
            return jsonify_error("Permission denied", status=403, code=ErrorCode.PERMISSION_DENIED)

        _logger.info("Config for dataset '%s' restored from history entry %d.", name, entry_id)

        # Rescan so the index picks up changes
        datasets.rescan_dataset(name)

        # Delete the restored entry so it doesn't clutter the list
        config_history.delete_entry(entry_id)

        return get_config(name)

    @app.delete("/configs/<name>")
    def delete_config(name: str):  # pyright: ignore[reportUnusedFunction]
        """Delete a dataset config and unregister the dataset."""
        if not datasets.unregister_dataset(name):
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        _logger.info("Config for dataset '%s' deleted.", name)
        return jsonify({"status": "ok"})
