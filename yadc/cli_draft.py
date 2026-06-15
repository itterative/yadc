"""CLI commands for managing caption drafts."""

import sys
from pathlib import Path
from typing import TextIO

import click

from yadc.cmd import status as cmd_status
from yadc.core import logging
from yadc.core.config import parse_config
from yadc.core.dataset import DatasetImage
from yadc.core.dataset_resolver import resolve_dataset
from yadc.utils.dict_utils import load_toml_file

from . import cli_common

_logger = logging.get_logger(__name__)


def _load_images(dataset_stream: TextIO):
    """Load and resolve dataset images from a config file."""
    dataset_toml_raw = load_toml_file(dataset_stream)

    try:
        dataset_toml = parse_config(dataset_toml_raw, strict=False)
    except Exception as e:
        _logger.error("Error loading dataset: %s", e)
        sys.exit(cmd_status.STATUS_ERROR)

    base_dir: str | None = None
    if dataset_stream.name:
        base_dir = str(Path(dataset_stream.name).parent)

    resolved_images: list[DatasetImage] = resolve_dataset(
        dataset_toml.dataset,
        dataset_toml.caption_suffix,
        base_dir=base_dir,
    )

    if not resolved_images:
        _logger.info("No images found in dataset.")
        sys.exit(cmd_status.STATUS_OK)

    return resolved_images


# --- draft command group ---


@click.group(
    "draft",
    short_help="Manage caption drafts",
    help="Manage caption drafts — save, list, show, or remove named drafts for dataset images.",
)
def draft():
    pass


# --- save ---


@draft.command(
    short_help="Save current captions as drafts",
    help=(
        "Save existing caption files as named drafts for later use. "
        "Reads each image's current caption and writes it to a draft file "
        "(e.g. image_name.<draft>.draft~). "
        "This is useful for backing up captions before re-captioning or for "
        "preserving multiple versions."
    ),
)
@click.argument("dataset", type=click.File("r"))
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name for the draft (e.g. 'v1', 'before-refinement').",
)
@click.option(
    "--overwrite/--no-overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing draft files.",
)
@click.option(
    "--skip-missing/--no-skip-missing",
    is_flag=True,
    default=True,
    help="Skip images that have no caption file. Default: skip.",
)
@cli_common.log_level
def save(
    dataset: TextIO,
    name: str,
    overwrite: bool,
    skip_missing: bool,
) -> None:
    _logger.info("Saving captions as draft: %s", name)

    resolved_images = _load_images(dataset)

    saved = 0
    skipped = 0

    for image in resolved_images:
        caption = image.read_caption()

        if not caption:
            if skip_missing:
                skipped += 1
                continue
            _logger.warning("No caption for %s", image.path)
            skipped += 1
            continue

        draft_file = image.draft_path(name)

        if draft_file.exists() and not overwrite:
            _logger.warning("Draft already exists for %s, skipping. Use --overwrite to replace.", image.path)
            skipped += 1
            continue

        image.write_draft(name, caption)
        _logger.debug("Saved draft for %s", image.path)
        saved += 1

    _logger.info("Saved %d drafts as '%s'. Skipped %d images.", saved, name, skipped)
    sys.exit(cmd_status.STATUS_OK)


# --- list ---


@draft.command(
    "list",
    short_help="List drafts for dataset images",
    help=("List all named drafts found alongside dataset images. Shows each image with its available draft names."),
)
@click.argument("dataset", type=click.File("r"))
@cli_common.log_level
def list_(
    dataset: TextIO,
) -> None:
    resolved_images = _load_images(dataset)

    total_drafts = 0

    for image in resolved_images:
        drafts = image.read_all_drafts()
        if not drafts:
            continue

        total_drafts += len(drafts)
        draft_names = ", ".join(drafts.keys())
        _logger.info("%s: %s", image.path, draft_names)

    if total_drafts == 0:
        _logger.info("No drafts found.")
    else:
        _logger.info("Found %d drafts across %d images.", total_drafts, sum(1 for img in resolved_images if img.read_all_drafts()))

    sys.exit(cmd_status.STATUS_OK)


# --- show ---


@draft.command(
    short_help="Show draft contents",
    help=("Print the contents of a named draft for each image in the dataset."),
)
@click.argument("dataset", type=click.File("r"))
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the draft to show.",
)
@cli_common.log_level
def show(
    dataset: TextIO,
    name: str,
) -> None:
    resolved_images = _load_images(dataset)

    shown = 0

    for image in resolved_images:
        content = image.read_draft(name)
        if not content:
            continue

        shown += 1
        click.echo(f"--- {image.path} ---")
        click.echo(content)
        click.echo("")

    if shown == 0:
        _logger.info("No draft '%s' found.", name)

    sys.exit(cmd_status.STATUS_OK)


# --- remove ---


@draft.command(
    short_help="Remove named drafts",
    help=("Delete a named draft file for each image in the dataset."),
)
@click.argument("dataset", type=click.File("r"))
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the draft to remove.",
)
@cli_common.log_level
def remove(
    dataset: TextIO,
    name: str,
) -> None:
    resolved_images = _load_images(dataset)

    removed = 0

    for image in resolved_images:
        draft_file = image.draft_path(name)
        if draft_file.exists():
            draft_file.unlink()
            _logger.debug("Removed draft for %s", image.path)
            removed += 1

    _logger.info("Removed %d drafts named '%s'.", removed, name)
    sys.exit(cmd_status.STATUS_OK)
