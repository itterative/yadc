"""sd-scripts export backend.

Supports the three metadata formats documented at:
https://github.com/itterative/sd-scripts/blob/docs/dataset_metadata_guide.md

- **json**  — ``metadata.json`` (JSON dict keyed by filename)
- **jsonl** — ``metadata.jsonl`` (one JSON object per line)
- **txt**   — Per-image caption files (DreamBooth method)
"""

import json
import pathlib
from dataclasses import dataclass
from typing import Any, cast

from ..dataset import DatasetImage
from .utils import read_caption_source


@dataclass(frozen=True)
class Backend:
    """Describes the sd-scripts export backend."""

    name: str = "sd-scripts"
    description: str = "sd-scripts metadata formats (metadata.json, metadata.jsonl, per-image caption files)"
    formats: tuple[str, ...] = ("json", "jsonl", "txt")


BACKEND = Backend()


def run(
    images: list[DatasetImage],
    *,
    fmt: str,
    source: str,
    drafts: tuple[str, ...] = (),
    output: pathlib.Path | None = None,
    append: bool = False,
    caption_extension: str = ".txt",
) -> int:
    """Run the sd-scripts export.

    Args:
        images: Resolved dataset images.
        fmt: One of ``'json'``, ``'jsonl'``, ``'txt'``.
        source: ``'caption'`` or ``'draft'``.
        drafts: Draft names. When source=``'draft'``, the first element is the
                primary draft (required). All drafts are appended after the
                primary source in order.
        output: Output file path (json/jsonl) or directory (txt).
                ``None`` means use defaults (metadata file alongside images
                or write caption files next to source images).
        append: Append to existing output.
        caption_extension: File extension for txt format (default: ``.caption``).

    Returns:
        Number of entries/files written.
    """
    if fmt == "json":
        return _export_json(images, output, source, drafts, append)
    elif fmt == "jsonl":
        return _export_jsonl(images, output, source, drafts, append)
    elif fmt == "txt":
        return _export_txt(images, source, drafts, output, caption_extension, append)
    else:
        raise ValueError(f"Unknown format: {fmt}")


# ---- internal helpers ----


def _export_json(
    images: list[DatasetImage],
    output_path: pathlib.Path | None,
    source: str,
    drafts: tuple[str, ...],
    append: bool,
) -> int:
    if output_path is None:
        raise ValueError("output_path is required for json format")

    existing: dict[str, dict[str, str]] = {}
    if append and output_path.exists():
        try:
            with open(output_path, "r") as f:
                loaded: Any = json.load(f)
                if isinstance(loaded, dict):
                    existing = cast(dict[str, Any], loaded)
        except (json.JSONDecodeError, ValueError):
            pass

    count = 0
    for image in images:
        try:
            text = read_caption_source(image, source, drafts)
        except FileNotFoundError:
            continue
        if not text:
            continue

        existing[image.absolute_path.name] = {"caption": text}
        count += 1

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
        _ = f.write("\n")

    return count


def _export_jsonl(
    images: list[DatasetImage],
    output_path: pathlib.Path | None,
    source: str,
    drafts: tuple[str, ...],
    append: bool,
) -> int:
    if output_path is None:
        raise ValueError("output_path is required for jsonl format")

    mode = "a" if append else "w"
    count = 0

    with open(output_path, mode) as f:
        for image in images:
            try:
                text = read_caption_source(image, source, drafts)
            except FileNotFoundError:
                continue
            if not text:
                continue

            entry = {"image_path": image.absolute_path.name, "caption": text}
            _ = f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    return count


def _export_txt(
    images: list[DatasetImage],
    source: str,
    drafts: tuple[str, ...],
    output_dir: pathlib.Path | None,
    caption_extension: str,
    append: bool,
) -> int:
    count = 0
    for image in images:
        try:
            text = read_caption_source(image, source, drafts)
        except FileNotFoundError:
            continue
        if not text:
            continue

        if output_dir is not None:
            out_path = output_dir / (image.absolute_path.stem + caption_extension)
        else:
            out_path = image.absolute_path.with_suffix(caption_extension)

        mode = "a" if append else "w"
        with open(out_path, mode) as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")

        count += 1

    return count
