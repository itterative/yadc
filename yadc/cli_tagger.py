"""``yadc tagger`` — start the tagger process or run a one-shot tagging job."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import click

from yadc.cmd.tagger import build_tagger_kwargs, resolve_label_path, tag_image
from yadc.taggers.base import TaggerResult


def _print_result(result: TaggerResult) -> None:
    """Pretty-print a TaggerResult, grouped by category when available."""
    if result.categories:
        for cat_name in ("rating", "general", "character"):
            cat_tags = result.categories.get(cat_name)
            if not cat_tags:
                continue
            click.echo(f"[{cat_name}]")
            for tag in cat_tags:
                score = result.tags.get(tag, 0.0)
                click.echo(f"  {tag}: {score:.4f}")
        # Any categories we didn't know how to print
        other = set(result.categories) - {"rating", "general", "character"}
        for cat_name in other:
            click.echo(f"[{cat_name}]")
            for tag in result.categories[cat_name]:
                score = result.tags.get(tag, 0.0)
                click.echo(f"  {tag}: {score:.4f}")
    else:
        for tag, score in result.tags.items():
            click.echo(f"{tag}: {score:.4f}")


def _profile_kwargs(preproc_profile: str, default_size: int) -> dict[str, Any]:
    """Build OnnxTagger kwargs for a preprocessing profile + optional size override.

    Validates the profile name (raises ``ValueError`` on an unknown value)
    so a typo in a CLI flag surfaces a clean error rather than being
    silently ignored.
    """
    from yadc.taggers.onnx_preprocess import get_profile

    kwargs: dict[str, Any] = {"preproc_profile": get_profile(preproc_profile)}
    if default_size > 0:
        kwargs["default_size"] = default_size
    return kwargs


@click.group("tagger")
def tagger():
    """Manage the ONNX tagger subprocess."""


@tagger.command()
@click.option("--model", "model_path", required=True, help="Path to the ONNX model file.")
@click.option(
    "--labels",
    "label_path",
    default=None,
    help="Path to a labels file (.csv for SmilingWolf, .txt for flat). Defaults to <model_dir>/selected_tags.csv.",
)
@click.option(
    "--preproc-profile",
    "preproc_profile",
    default="wd-tagger",
    type=click.Choice(["wd-tagger", "timm", "pytorch"]),
    help="Preprocessing profile: 'wd-tagger' (SmilingWolf convention; default) or 'timm' (PyTorch / timm convention with NCHW + RGB + ImageNet).",
)
@click.option(
    "--default-size",
    "default_size",
    default=0,
    type=int,
    show_default=True,
    help="Override the profile's default input size when the model has symbolic H/W dims (e.g. 512 for animetimm ConvNeXt). 0 means use the profile default.",
)
def start(model_path: str, label_path: str | None, preproc_profile: str, default_size: int) -> None:
    """Start the tagger process and keep it running (for debugging)."""
    from yadc.taggers.onnx import OnnxTagger
    from yadc.taggers.server import TaggerServer

    resolved = resolve_label_path(model_path, label_path)
    kwargs = build_tagger_kwargs(resolved)
    kwargs.update(_profile_kwargs(preproc_profile, default_size))
    click.echo(f"Starting tagger with model: {model_path}")
    if resolved:
        click.echo(f"Labels: {resolved}")
    click.echo(f"Preproc profile: {preproc_profile} (default_size={default_size or 'profile'})")

    server = TaggerServer(OnnxTagger, model_path, kwargs)
    server.start()
    click.echo("Tagger is ready. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nShutting down tagger...")
        server.stop()


@tagger.command()
@click.option("--model", "model_path", required=True, help="Path to the ONNX model file.")
@click.option(
    "--labels",
    "label_path",
    default=None,
    help="Path to a labels file (.csv for SmilingWolf, .txt for flat). Defaults to <model_dir>/selected_tags.csv.",
)
@click.option(
    "--preproc-profile",
    "preproc_profile",
    default="wd-tagger",
    type=click.Choice(["wd-tagger", "timm", "pytorch"]),
    help="Preprocessing profile: 'wd-tagger' (SmilingWolf convention; default) or 'timm' (PyTorch / timm convention with NCHW + RGB + ImageNet).",
)
@click.option(
    "--default-size",
    "default_size",
    default=0,
    type=int,
    show_default=True,
    help="Override the profile's default input size when the model has symbolic H/W dims (e.g. 512 for animetimm ConvNeXt). 0 means use the profile default.",
)
@click.argument("image", type=click.Path(exists=True))
def tag(model_path: str, label_path: str | None, preproc_profile: str, default_size: int, image: str) -> None:
    """Tag a single image and print the categorized result."""
    from yadc.taggers.onnx import OnnxTagger

    resolved = resolve_label_path(model_path, label_path)
    kwargs = build_tagger_kwargs(resolved)
    kwargs.update(_profile_kwargs(preproc_profile, default_size))
    image_bytes = Path(image).read_bytes()
    result = tag_image(OnnxTagger, model_path, image_bytes, kwargs)
    _print_result(result)
