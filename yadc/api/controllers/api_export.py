"""Export endpoints — trigger caption exports to training-tool compatible formats."""

import pathlib
from typing import ClassVar, Literal

import pydantic
from quart import jsonify, request

from yadc.api.services.datasets import DatasetService
from yadc.core.config import parse_config
from yadc.core.dataset_resolver import resolve_dataset
from yadc.core.exporters import get_backend, list_backends, run_export
from yadc.utils.dict_utils import load_toml_file, toml_to_plain

from ..modules.logging_factory import LoggingFactory
from . import controller
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_error, validate_body


class ExportBody(pydantic.BaseModel):
    dataset: str
    backend: str = "sd-scripts"
    format: str | None = None
    source: Literal["caption", "draft"] | None = None
    draft: str | None = None
    with_drafts: list[str] = pydantic.Field(default_factory=list)
    output: str | None = None
    append: bool = False
    caption_extension: str = ".txt"

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")

    @pydantic.model_validator(mode="after")
    def _resolve(self) -> "ExportBody":
        if self.source is None:
            self.source = "draft" if self.draft else "caption"
        if self.source == "draft" and not self.draft:
            raise ValueError("'draft' is required when source='draft'")
        if self.format is None:
            if self.output:
                ext = pathlib.Path(self.output).suffix.lower()
                self.format = {".json": "json", ".jsonl": "jsonl"}.get(ext, "txt")
            else:
                self.format = "jsonl"
        return self


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
            dataset (str):        Dataset name to export. Required.
            backend (str):        Export backend name. Default: "sd-scripts".
            format (str):         Output format (backend-specific). Default: inferred from
                                  ``output`` extension, or "jsonl" if ``output`` is unset.
            source (str):         "caption" or "draft". Default: derived from ``draft``
                                  ("draft" if set, else "caption"). When "draft",
                                  ``draft`` is required.
            draft (str):          Primary draft name. Required when ``source="draft"``.
            with_drafts (str[]):  Additional drafts to append.
            output (str):         Output file/dir path. Default: auto-detect.
            append (bool):        Append to existing output. Default: false.
            caption_extension (str): File extension for txt format. Default: ".txt".
        """
        body = validate_body(ExportBody, await request.get_json(silent=True))
        # ExportBody._resolve guarantees these are always set after validation.
        assert body.source is not None
        assert body.format is not None

        dataset_info = datasets.get_dataset(body.dataset)
        if dataset_info is None or dataset_info.config_path is None:
            return jsonify_error(f"Dataset '{body.dataset}' not found", status=404, code=ErrorCode.NOT_FOUND)

        try:
            backend_desc = get_backend(body.backend)
        except ValueError as e:
            return jsonify_error(str(e), status=400, code=ErrorCode.BAD_REQUEST)

        if body.format not in backend_desc.formats:
            return jsonify_error(
                f"Backend '{body.backend}' does not support format '{body.format}'. Available: {', '.join(backend_desc.formats)}",
                status=400,
                code=ErrorCode.BAD_REQUEST,
            )

        if body.source == "draft":
            assert body.draft is not None  # guaranteed by ExportBody._resolve
            drafts = (body.draft, *body.with_drafts)
        else:
            drafts = tuple(body.with_drafts)

        # --- Resolve images from the dataset config ---
        config_path = pathlib.Path(dataset_info.config_path)
        if not config_path.exists():
            return jsonify_error(f"Config file not found: {config_path}", status=404, code=ErrorCode.NOT_FOUND)

        try:
            with open(config_path) as f:
                raw = load_toml_file(f)
            config = parse_config(toml_to_plain(raw), strict=False)
        except Exception as e:
            return jsonify_error(f"Invalid dataset config: {e}", status=400, code=ErrorCode.BAD_REQUEST)

        images = resolve_dataset(config.dataset, config.caption_suffix, base_dir=str(config_path.parent))
        if not images:
            return jsonify_error("No images found in dataset", status=400, code=ErrorCode.BAD_REQUEST)

        # --- Output path ---
        output_path: pathlib.Path | None = None

        if body.output:
            output_path = pathlib.Path(body.output)
            if body.format == "txt":
                if not output_path.is_dir():
                    return jsonify_error("Output must be an existing directory for txt format", status=400, code=ErrorCode.BAD_REQUEST)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default: place metadata alongside the first image directory
            first_dir = images[0].absolute_path.parent
            if body.format == "json":
                output_path = first_dir / "metadata.json"
            elif body.format == "jsonl":
                output_path = first_dir / "metadata.jsonl"
            # txt: output_path stays None → writes alongside images

        # --- Run export ---
        try:
            count = run_export(
                body.backend,
                images,
                fmt=body.format,
                source=body.source,
                drafts=drafts,
                output=output_path,
                append=body.append,
                caption_extension=body.caption_extension,
            )
        except ValueError as e:
            return jsonify_error(str(e), status=400, code=ErrorCode.BAD_REQUEST)
        except Exception as e:
            _logger.exception("Export failed for dataset '%s': %s", body.dataset, e)
            return jsonify_error(str(e), status=500, code=ErrorCode.INTERNAL_ERROR)

        _logger.info(
            "Exported %d images from '%s' (backend=%s, format=%s, source=%s)",
            count,
            body.dataset,
            body.backend,
            body.format,
            body.source,
        )

        return jsonify(
            {
                "status": "ok",
                "count": count,
                "dataset": body.dataset,
                "backend": body.backend,
                "format": body.format,
                "source": body.source,
                "output": str(output_path) if output_path else "per-image",
            }
        )
