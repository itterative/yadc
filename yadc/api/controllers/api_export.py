"""Export endpoints — trigger caption exports to training-tool compatible formats."""

import pathlib
from typing import Any

from quart import jsonify, request

from yadc.api.services.datasets import DatasetService
from yadc.core.config import parse_config
from yadc.core.dataset_resolver import resolve_dataset
from yadc.core.exporters import get_backend, list_backends, run_export

from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_error


@controller
def api_export(app: ApiBlueprint, logging: LoggingFactory, datasets: DatasetService):
    _logger = logging.get_logger(__name__)

    @app.get("/export/backends")
    def list_export_backends():  # pyright: ignore[reportUnusedFunction]
        """List available export backends and their supported formats."""
        backends = list_backends()
        return jsonify(
            [
                {
                    "name": desc.name,
                    "description": desc.description,
                    "formats": list(desc.formats),
                }
                for desc in backends.values()
            ]
        )

    @app.post("/export")
    async def export_dataset():  # pyright: ignore[reportUnusedFunction]
        """Run an export for a dataset.

        JSON body:
            dataset (str):      Dataset name to export.
            backend (str):      Export backend name. Default: "sd-scripts".
            format (str):       Output format (backend-specific). Default: inferred.
            source (str):       "caption" or "draft". Default: "caption".
            draft (str?):       Primary draft name (required if source=draft).
            with_drafts (str[]?) Additional drafts to append.
            output (str?):      Output file/dir path. Default: auto-detect.
            append (bool):      Append to existing output. Default: false.
            caption_extension (str): File extension for txt format. Default: ".txt".
        """
        body: dict[str, Any] = await request.get_json(silent=True) or {}

        # --- Required fields ---
        dataset_name = body.get("dataset")
        if not dataset_name:
            return jsonify_error("'dataset' is required", status=400, code=ErrorCode.BAD_REQUEST)

        dataset_info = datasets.get_dataset(dataset_name)
        if dataset_info is None or dataset_info.config_path is None:
            return jsonify_error(f"Dataset '{dataset_name}' not found", status=404, code=ErrorCode.NOT_FOUND)

        # --- Resolve backend & format ---
        backend_name = body.get("backend", "sd-scripts")
        try:
            backend_desc = get_backend(backend_name)
        except ValueError as e:
            return jsonify_error(str(e), status=400, code=ErrorCode.BAD_REQUEST)

        fmt = body.get("format")
        if fmt is None:
            output = body.get("output")
            if output:
                ext = pathlib.Path(output).suffix.lower()
                fmt = {".json": "json", ".jsonl": "jsonl"}.get(ext, "txt")
            else:
                fmt = "jsonl"

        if fmt not in backend_desc.formats:
            return jsonify_error(
                f"Backend '{backend_name}' does not support format '{fmt}'. Available: {', '.join(backend_desc.formats)}",
                status=400,
                code=ErrorCode.BAD_REQUEST,
            )

        # --- Source & drafts ---
        draft_name = body.get("draft")
        with_drafts: list[str] = body.get("with_drafts", [])

        if draft_name is not None:
            source = "draft"
            drafts = tuple([draft_name, *with_drafts])
        else:
            source = "caption"
            drafts = tuple(with_drafts)

        # --- Resolve images from the dataset config ---
        import toml as toml_lib

        config_path = pathlib.Path(dataset_info.config_path)
        if not config_path.exists():
            return jsonify_error(f"Config file not found: {config_path}", status=404, code=ErrorCode.NOT_FOUND)

        try:
            with open(config_path) as f:
                raw = toml_lib.load(f)
            config = parse_config(raw, strict=False)
        except Exception as e:
            return jsonify_error(f"Invalid dataset config: {e}", status=400, code=ErrorCode.BAD_REQUEST)

        images = resolve_dataset(config.dataset, config.caption_suffix, base_dir=str(config_path.parent))
        if not images:
            return jsonify_error("No images found in dataset", status=400, code=ErrorCode.BAD_REQUEST)

        # --- Output path ---
        output_str: str | None = body.get("output")
        output_path: pathlib.Path | None = None

        if output_str:
            output_path = pathlib.Path(output_str)
            if fmt == "txt":
                if not output_path.is_dir():
                    return jsonify_error("Output must be an existing directory for txt format", status=400, code=ErrorCode.BAD_REQUEST)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default: place metadata alongside the first image directory
            first_dir = images[0].absolute_path.parent
            if fmt == "json":
                output_path = first_dir / "metadata.json"
            elif fmt == "jsonl":
                output_path = first_dir / "metadata.jsonl"
            # txt: output_path stays None → writes alongside images

        append = body.get("append", False)
        caption_extension = body.get("caption_extension", ".txt")

        # --- Run export ---
        try:
            count = run_export(
                backend_name,
                images,
                fmt=fmt,
                source=source,
                drafts=drafts,
                output=output_path,
                append=append,
                caption_extension=caption_extension,
            )
        except ValueError as e:
            return jsonify_error(str(e), status=400, code=ErrorCode.BAD_REQUEST)
        except Exception as e:
            _logger.exception("Export failed for dataset '%s': %s", dataset_name, e)
            return jsonify_error(str(e), status=500, code=ErrorCode.INTERNAL_ERROR)

        _logger.info(
            "Exported %d images from '%s' (backend=%s, format=%s, source=%s)",
            count,
            dataset_name,
            backend_name,
            fmt,
            source,
        )

        return jsonify(
            {
                "status": "ok",
                "count": count,
                "dataset": dataset_name,
                "backend": backend_name,
                "format": fmt,
                "source": source,
                "output": str(output_path) if output_path else "per-image",
            }
        )
