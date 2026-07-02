"""sd-scripts export backend.

Supports the three metadata formats documented at:
https://github.com/itterative/sd-scripts/blob/docs/dataset_metadata_guide.md

- **json**  — ``metadata.json`` (JSON dict keyed by filename)
- **jsonl** — ``metadata.jsonl`` (one JSON object per line)
- **txt**   — Per-image caption files (DreamBooth method)

Also supports exporting to a zip archive via :func:`run_zip`, which bundles
the metadata and (optionally) source images into a downloadable zip file.
"""

import json
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from ..dataset import DatasetImage
from .utils import read_caption_source, relative_arc_name
from .zip_stream import ZipMember, bytes_member, file_member


@dataclass(frozen=True)
class Backend:
    """Describes the sd-scripts export backend."""

    name: str = "sd-scripts"
    description: str = "sd-scripts metadata formats (metadata.json, metadata.jsonl, per-image caption files)"
    formats: tuple[str, ...] = ("json", "jsonl", "txt")
    zip_only: bool = False


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
    delimiter: str = "\n",
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
        delimiter: String used to join primary source and chained drafts when
                more than one segment contributes to the caption. Default ``"\\n"``.

    Returns:
        Number of entries/files written.
    """
    if fmt == "json":
        return _export_json(images, output, source, drafts, append, delimiter=delimiter)
    elif fmt == "jsonl":
        return _export_jsonl(images, output, source, drafts, append, delimiter=delimiter)
    elif fmt == "txt":
        return _export_txt(images, source, drafts, output, caption_extension, append, delimiter=delimiter)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def iter_zip_members(
    images: list[DatasetImage],
    *,
    fmt: str,
    source: str,
    drafts: tuple[str, ...] = (),
    caption_extension: str = ".txt",
    include_images: bool = False,
    base_dir: pathlib.Path | None = None,
    delimiter: str = "\n",
) -> tuple[Iterator[ZipMember], int]:
    """Build the zip's entries for the sd-scripts export.

    Pre-flight validation runs eagerly here (captions are read, image files are
    stated) so a missing file raises before any bytes are streamed. The returned
    member iterator is lazy — image bytes are read on demand as the zip is written.

    Returns:
        ``(members, count)`` — the zip entries and the number of captioned entries.
    """
    if base_dir is None:
        base_dir = images[0].absolute_path.parent.parent

    now = datetime.now()
    members: list[ZipMember] = []

    if fmt == "json":
        data, count = _build_json(images, source, drafts, base_dir, delimiter=delimiter)
        members.append(bytes_member("metadata.json", data, now))
    elif fmt == "jsonl":
        data, count = _build_jsonl(images, source, drafts, base_dir, delimiter=delimiter)
        members.append(bytes_member("metadata.jsonl", data, now))
    elif fmt == "txt":
        captions, count = _build_txt(images, source, drafts, caption_extension, base_dir, delimiter=delimiter)
        members.extend(bytes_member(name, text, now) for name, text in captions)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if include_images:
        _append_image_members(members, images, base_dir)

    return iter(members), count


# ---- internal helpers ----


def _collect_captions(
    images: list[DatasetImage],
    source: str,
    drafts: tuple[str, ...],
    *,
    delimiter: str = "\n",
) -> list[tuple[DatasetImage, str]]:
    """Collect (image, caption_text) pairs for all images with valid captions."""
    result: list[tuple[DatasetImage, str]] = []
    for image in images:
        try:
            text = read_caption_source(image, source, drafts, delimiter=delimiter)
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
    *,
    delimiter: str = "\n",
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
            text = read_caption_source(image, source, drafts, delimiter=delimiter)
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


def _build_json(
    images: list[DatasetImage],
    source: str,
    drafts: tuple[str, ...],
    base_dir: pathlib.Path,
    *,
    delimiter: str = "\n",
) -> tuple[bytes, int]:
    data: dict[str, dict[str, str]] = {}
    count = 0
    for image, text in _collect_captions(images, source, drafts, delimiter=delimiter):
        data[relative_arc_name(image.absolute_path, base_dir)] = {"caption": text}
        count += 1
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"), count


def _build_jsonl(
    images: list[DatasetImage],
    source: str,
    drafts: tuple[str, ...],
    base_dir: pathlib.Path,
    *,
    delimiter: str = "\n",
) -> tuple[bytes, int]:
    lines: list[str] = []
    count = 0
    for image, text in _collect_captions(images, source, drafts, delimiter=delimiter):
        entry = {"image_path": relative_arc_name(image.absolute_path, base_dir), "caption": text}
        lines.append(json.dumps(entry, ensure_ascii=False))
        count += 1
    body = ("\n".join(lines) + "\n") if lines else ""
    return body.encode("utf-8"), count


def _export_jsonl(
    images: list[DatasetImage],
    output_path: pathlib.Path | None,
    source: str,
    drafts: tuple[str, ...],
    append: bool,
    *,
    delimiter: str = "\n",
) -> int:
    if output_path is None:
        raise ValueError("output_path is required for jsonl format")

    mode = "a" if append else "w"
    count = 0

    with open(output_path, mode) as f:
        for image in images:
            try:
                text = read_caption_source(image, source, drafts, delimiter=delimiter)
            except FileNotFoundError:
                continue
            if not text:
                continue

            entry = {"image_path": image.absolute_path.name, "caption": text}
            _ = f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    return count


def _build_txt(
    images: list[DatasetImage],
    source: str,
    drafts: tuple[str, ...],
    caption_extension: str,
    base_dir: pathlib.Path,
    *,
    delimiter: str = "\n",
) -> tuple[list[tuple[str, bytes]], int]:
    out: list[tuple[str, bytes]] = []
    count = 0
    for image, text in _collect_captions(images, source, drafts, delimiter=delimiter):
        rel = relative_arc_name(image.absolute_path, base_dir)
        caption_rel = str(pathlib.PurePosixPath(rel).with_suffix(caption_extension))
        content = text if text.endswith("\n") else text + "\n"
        out.append((caption_rel, content.encode("utf-8")))
        count += 1
    return out, count


def _append_image_members(members: list[ZipMember], images: list[DatasetImage], base_dir: pathlib.Path) -> None:
    """Append STORED members for each distinct image file (deduplicated by arc name)."""
    seen: set[str] = set()
    for image in images:
        img_path = image.absolute_path
        arc_name = relative_arc_name(img_path, base_dir)
        if arc_name in seen:
            continue
        seen.add(arc_name)
        st = img_path.stat()  # raises here (pre-flight) if the image is missing
        members.append(file_member(arc_name, img_path, datetime.fromtimestamp(st.st_mtime), st.st_mode))


def _export_txt(
    images: list[DatasetImage],
    source: str,
    drafts: tuple[str, ...],
    output_dir: pathlib.Path | None,
    caption_extension: str,
    append: bool,
    *,
    delimiter: str = "\n",
) -> int:
    count = 0
    for image in images:
        try:
            text = read_caption_source(image, source, drafts, delimiter=delimiter)
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
