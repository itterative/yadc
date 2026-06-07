"""sd-scripts export backend.

Supports the three metadata formats documented at:
https://github.com/itterative/sd-scripts/blob/docs/dataset_metadata_guide.md

- **json**  — ``metadata.json`` (JSON dict keyed by filename)
- **jsonl** — ``metadata.jsonl`` (one JSON object per line)
- **txt**   — Per-image caption files (DreamBooth method)

Also supports exporting to a zip archive via :func:`run_zip`, which bundles
the metadata and (optionally) source images into a downloadable zip file.
"""

import io
import json
import pathlib
import zipfile
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


def _image_zip_path(image: DatasetImage, base_dir: pathlib.Path) -> str:
    """Compute the relative path for an image inside a zip archive.

    Falls back to the bare filename if the image is not under *base_dir*
    (e.g. external / imported datasets).
    """
    try:
        return str(image.absolute_path.relative_to(base_dir))
    except ValueError:
        return image.absolute_path.name


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


def run_zip(
    images: list[DatasetImage],
    *,
    fmt: str,
    source: str,
    drafts: tuple[str, ...] = (),
    caption_extension: str = ".txt",
    include_images: bool = False,
    base_dir: pathlib.Path | None = None,
) -> tuple[io.BytesIO, int]:
    """Run the sd-scripts export into an in-memory zip archive.

    Args:
        images: Resolved dataset images.
        fmt: One of ``'json'``, ``'jsonl'``, ``'txt'``.
        source: ``'caption'`` or ``'draft'``.
        drafts: Draft names to include.
        caption_extension: File extension for txt format.
        include_images: Whether to include source image files in the zip.
        base_dir: Dataset base directory (parent of config.toml). Used to
                  compute relative paths inside the zip. Falls back to the
                  parent of the first image's directory.

    Returns:
        ``(zip_bytes, count)`` — the zip as a BytesIO buffer and the number
        of exported entries.
    """
    if base_dir is None:
        base_dir = images[0].absolute_path.parent.parent

    buf = io.BytesIO()
    count = 0

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if fmt == "json":
            count = _zip_json(zf, images, source, drafts, base_dir)
        elif fmt == "jsonl":
            count = _zip_jsonl(zf, images, source, drafts, base_dir)
        elif fmt == "txt":
            count = _zip_txt(zf, images, source, drafts, caption_extension, base_dir)
        else:
            raise ValueError(f"Unknown format: {fmt}")

        if include_images:
            _zip_images(zf, images, base_dir)

    buf.seek(0)
    return buf, count


# ---- internal helpers ----


def _collect_captions(
    images: list[DatasetImage],
    source: str,
    drafts: tuple[str, ...],
) -> list[tuple[DatasetImage, str]]:
    """Collect (image, caption_text) pairs for all images with valid captions."""
    result: list[tuple[DatasetImage, str]] = []
    for image in images:
        try:
            text = read_caption_source(image, source, drafts)
        except FileNotFoundError:
            continue
        if not text:
            continue
        result.append((image, text))
    return result


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


def _zip_json(
    zf: zipfile.ZipFile,
    images: list[DatasetImage],
    source: str,
    drafts: tuple[str, ...],
    base_dir: pathlib.Path,
) -> int:
    data: dict[str, dict[str, str]] = {}
    count = 0
    for image, text in _collect_captions(images, source, drafts):
        data[_image_zip_path(image, base_dir)] = {"caption": text}
        count += 1

    zf.writestr("metadata.json", json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return count


def _zip_jsonl(
    zf: zipfile.ZipFile,
    images: list[DatasetImage],
    source: str,
    drafts: tuple[str, ...],
    base_dir: pathlib.Path,
) -> int:
    lines: list[str] = []
    count = 0
    for image, text in _collect_captions(images, source, drafts):
        entry = {"image_path": _image_zip_path(image, base_dir), "caption": text}
        lines.append(json.dumps(entry, ensure_ascii=False))
        count += 1

    zf.writestr("metadata.jsonl", "\n".join(lines) + ("\n" if lines else ""))
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


def _zip_txt(
    zf: zipfile.ZipFile,
    images: list[DatasetImage],
    source: str,
    drafts: tuple[str, ...],
    caption_extension: str,
    base_dir: pathlib.Path,
) -> int:
    count = 0
    for image, text in _collect_captions(images, source, drafts):
        rel = _image_zip_path(image, base_dir)
        caption_rel = pathlib.PurePosixPath(rel).with_suffix(caption_extension)
        content = text if text.endswith("\n") else text + "\n"
        zf.writestr(str(caption_rel), content)
        count += 1
    return count


def _zip_images(
    zf: zipfile.ZipFile,
    images: list[DatasetImage],
    base_dir: pathlib.Path,
) -> None:
    """Add all image files to the zip archive."""
    seen: set[str] = set()
    for image in images:
        arc_name = _image_zip_path(image, base_dir)
        if arc_name in seen:
            continue
        seen.add(arc_name)
        zf.write(image.absolute_path, arc_name)


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
