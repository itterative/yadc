"""CLI command for exporting dataset captions to training-tool compatible formats."""

import pathlib
import sys
from typing import TextIO

import click

from yadc.cmd import status as cmd_status
from yadc.utils.dict_utils import load_toml_file

from . import cli_common
from .core import logging
from .core.config import parse_config
from .core.dataset import DatasetImage
from .core.dataset_resolver import resolve_dataset
from .core.exporters import list_backends, run_export

_logger = logging.get_logger(__name__)

_BACKEND_NAMES = list(list_backends().keys())
_EXPORT_MAP = {".json": "json", ".jsonl": "jsonl"}


@click.command(
    short_help="Export captions for training tools",
    help="Export caption results to formats compatible with training tools. Backends: " + ", ".join(f"{name}" for name in _BACKEND_NAMES) + ".",
)
@click.argument("dataset", type=click.File("r"))
@click.option(
    "--backend",
    "-b",
    type=click.Choice(_BACKEND_NAMES),
    default="sd-scripts",
    help="Export backend. Default: sd-scripts.",
)
@click.option(
    "--draft",
    type=str,
    default=None,
    help="Export a named draft instead of the caption file.",
)
@click.option(
    "--with-draft",
    "with_drafts",
    type=str,
    multiple=True,
    help="Append a named draft after the primary source. May be repeated.",
)
@click.option(
    "--format",
    "fmt",
    type=str,
    default=None,
    help="Output format (backend-specific). sd-scripts: json, jsonl, txt. Default: inferred from --output extension, or jsonl.",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path (for json/jsonl) or directory (for txt). Default: metadata file alongside images, or per-image caption files.",
)
@click.option(
    "--append/--overwrite",
    is_flag=True,
    default=False,
    help="Append to existing output instead of overwriting.",
)
@click.option(
    "--caption-extension",
    type=str,
    default=".txt",
    help="File extension for per-image caption files (txt format). Default: .txt.",
)
@click.option("--env", type=str, default=None, help="Configuration environment")
@click.option("--user-config", type=str, default=None, help="Base user config")
@cli_common.log_level
def export(
    dataset: TextIO,
    backend: str,
    draft: str | None,
    with_drafts: tuple[str, ...],
    fmt: str | None,
    output: str | None,
    append: bool,
    caption_extension: str,
    env: str | None,  # pyright: ignore[reportUnusedParameter]
    user_config: str | None,  # pyright: ignore[reportUnusedParameter]
) -> None:
    backend_name = backend
    if draft is not None:
        source = "draft"
        drafts = (draft, *with_drafts)
    else:
        source = "caption"
        drafts = with_drafts

    dataset_toml_raw = load_toml_file(dataset)

    try:
        dataset_toml = parse_config(dataset_toml_raw)
    except Exception as e:
        _logger.error("Error loading dataset: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    resolved_images: list[DatasetImage] = resolve_dataset(
        dataset_toml.dataset,
        dataset_toml.caption_suffix,
    )

    if not resolved_images:
        _logger.info("No images found in dataset.")
        sys.exit(cmd_status.STATUS_OK)

    # resolve backend and validate format
    from .core.exporters import get_backend

    try:
        backend_desc = get_backend(backend_name)
    except ValueError as e:
        _logger.error("Error: %s", e)
        sys.exit(cmd_status.STATUS_USER_ERROR)

    # infer format from output extension if not specified
    if fmt is None:
        if output is not None:
            ext = pathlib.Path(output).suffix.lower()
            fmt = _EXPORT_MAP.get(ext, "txt")
        else:
            fmt = "jsonl"

    if fmt not in backend_desc.formats:
        _logger.error(
            "Error: backend %r does not support format %r. Available: %s",
            backend_name,
            fmt,
            ", ".join(backend_desc.formats),
        )
        sys.exit(cmd_status.STATUS_USER_ERROR)

    # determine output path
    output_path: pathlib.Path | None = None

    if output is not None:
        if fmt == "txt":
            output_path = pathlib.Path(output)
            if not output_path.is_dir():
                _logger.error("Error: --output must be an existing directory for txt format")
                sys.exit(cmd_status.STATUS_USER_ERROR)
        else:
            output_path = pathlib.Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        first_dir = resolved_images[0].absolute_path.parent
        if fmt == "json":
            output_path = first_dir / "metadata.json"
        elif fmt == "jsonl":
            output_path = first_dir / "metadata.jsonl"
        # txt: output_path stays None → writes alongside images

    _logger.info(
        "Exporting %d images (backend=%s, source=%s, format=%s)",
        len(resolved_images),
        backend_name,
        source,
        fmt,
    )

    if output_path and fmt != "txt":
        _logger.info("Output: %s", output_path)
    elif output_path and fmt == "txt":
        _logger.info("Output directory: %s", output_path)
    else:
        _logger.info("Writing caption files alongside images.")

    try:
        count = run_export(
            backend_name,
            resolved_images,
            fmt=fmt,
            source=source,
            drafts=drafts,
            output=output_path,
            append=append,
            caption_extension=caption_extension,
        )
    except Exception as e:
        _logger.error("Error: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    _logger.info("Exported %d captions.", count)
    sys.exit(cmd_status.STATUS_OK)
