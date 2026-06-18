"""yadc export backend — raw archive of images and their sidecar files.

Unlike ``sd-scripts`` (which transforms captions into training metadata), this
backend bundles each image together with its on-disk sidecars — caption,
drafts, metadata TOML, TOML backup, and history — preserving yadc's native
file layout. The result is a faithful, round-trippable zip of the dataset.

The format *is* a zip, so the backend is zip-only: :func:`run` raises. Source /
draft selection and ``include_images`` are ignored, since every sidecar is
archived as-is.
"""

import pathlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

from ..dataset import DatasetImage
from .utils import relative_arc_name
from .zip_stream import ZipMember, file_member


@dataclass(frozen=True)
class Backend:
    """Describes the yadc export backend."""

    name: str = "yadc"
    description: str = "Raw zip of images and their sidecar files (caption, drafts, metadata TOML, history)"
    formats: tuple[str, ...] = ("zip",)
    zip_only: bool = True


BACKEND = Backend()


def run(images: list[DatasetImage], **_kwargs: object) -> int:  # pyright: ignore[reportUnusedParameter]
    """Not supported — the yadc backend exports zips only.

    Raises:
        ValueError: Always. Use :func:`stream_export_zip` (the Web UI export
            dialog does this automatically). The CLI does not currently support zip export.
    """
    raise ValueError(
        "yadc backend exports zips only — use stream_export_zip() (the Web UI export "
        "dialog does this automatically). The CLI does not currently support zip export."
    )


def iter_zip_members(
    images: list[DatasetImage],
    *,
    base_dir: pathlib.Path | None = None,
    **_kwargs: object,
) -> tuple[Iterator[ZipMember], int]:
    """Build the zip's entries — every image plus all of its sidecar files.

    Pre-flight validation runs eagerly here (each file is stated) so a missing
    file raises before any bytes are streamed. The returned member iterator is
    lazy — file bytes are read on demand as the zip is written. Images are always
    included (this backend is a raw archive), so *include_images* is accepted for
    signature compatibility but ignored.

    Returns:
        ``(members, image_count)`` — the zip entries and the number of images archived.
    """
    if base_dir is None:
        base_dir = images[0].absolute_path.parent.parent

    members: list[ZipMember] = []
    seen: set[str] = set()
    count = 0

    for image in images:
        if _append_with_sidecars(members, image, base_dir, seen):
            count += 1

    return iter(members), count


# ---- internal helpers ----


def _sidecar_files(image: DatasetImage) -> list[pathlib.Path]:
    """All sidecar files for *image* — files sharing its stem, minus the image itself.

    Globbing ``<stem>.*`` (rather than enumerating suffixes) captures every
    sidecar type — caption ``.txt``, drafts ``.<name>.draft~``, metadata
    ``.toml``, backup ``.toml~``, history ``.history~`` — and any future ones.
    The literal ``.`` after the stem avoids matching a longer filename such as
    ``img0010.png`` for an ``img001`` image.
    """
    img_path = image.absolute_path
    return [p for p in img_path.parent.glob(img_path.stem + ".*") if p.is_file() and p != img_path]


def _append_with_sidecars(
    members: list[ZipMember],
    image: DatasetImage,
    base_dir: pathlib.Path,
    seen: set[str],
) -> bool:
    """Append STORED members for the image and its sidecars, deduplicated by arc name.

    Returns True when the image itself was newly added. Sidecars shared across two
    dataset entries are not added twice. Each file is stated up front (pre-flight)
    so a missing file raises before streaming begins.
    """
    img_path = image.absolute_path
    img_arc = relative_arc_name(img_path, base_dir)
    added = False
    if img_arc not in seen:
        seen.add(img_arc)
        st = img_path.stat()
        members.append(file_member(img_arc, img_path, datetime.fromtimestamp(st.st_mtime), st.st_mode))
        added = True

    for sidecar in _sidecar_files(image):
        arc = relative_arc_name(sidecar, base_dir)
        if arc in seen:
            continue
        seen.add(arc)
        st = sidecar.stat()
        members.append(file_member(arc, sidecar, datetime.fromtimestamp(st.st_mtime), st.st_mode))

    return added
