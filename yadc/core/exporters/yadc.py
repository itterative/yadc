"""yadc export backend — raw archive of images and their sidecar files.

Unlike ``sd-scripts`` (which transforms captions into training metadata), this
backend bundles each image together with its on-disk sidecars — caption,
drafts, metadata TOML, TOML backup, and history — preserving yadc's native
file layout. The result is a faithful, round-trippable zip of the dataset.

The format *is* a zip, so the backend is zip-only: :func:`run` raises. Source /
draft selection and ``include_images`` are ignored, since every sidecar is
archived as-is.
"""

import io
import pathlib
import zipfile
from dataclasses import dataclass

from ..dataset import DatasetImage
from .utils import relative_arc_name


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
        ValueError: Always. Use :func:`run_zip` (the Web UI export dialog calls
            it automatically). The CLI has no zip support yet.
    """
    raise ValueError(
        "yadc backend exports zips only — use run_export_zip() (the Web UI export "
        "dialog does this automatically). The CLI does not currently support zip export."
    )


def run_zip(
    images: list[DatasetImage],
    *,
    base_dir: pathlib.Path | None = None,
    **_kwargs: object,
) -> tuple[io.BytesIO, int]:
    """Bundle every image and all of its sidecar files into an in-memory zip.

    Images are always included (this backend is a raw archive), so
    *include_images* is accepted for signature compatibility but ignored.

    Returns:
        ``(zip_bytes, image_count)`` — the zip as a BytesIO buffer and the
        number of images archived.
    """
    if base_dir is None:
        base_dir = images[0].absolute_path.parent.parent

    buf = io.BytesIO()
    seen: set[str] = set()
    count = 0

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for image in images:
            if _add_with_sidecars(zf, image, base_dir, seen):
                count += 1

    buf.seek(0)
    return buf, count


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


def _add_with_sidecars(
    zf: zipfile.ZipFile,
    image: DatasetImage,
    base_dir: pathlib.Path,
    seen: set[str],
) -> bool:
    """Write the image and its sidecars into *zf* under their relative paths.

    Returns True when the image itself was newly added. Entries are deduplicated
    by arc name so a sidecar shared across two dataset entries isn't written twice.
    """
    img_path = image.absolute_path
    img_arc = relative_arc_name(img_path, base_dir)
    added = False
    if img_arc not in seen:
        seen.add(img_arc)
        zf.write(img_path, img_arc)
        added = True

    for sidecar in _sidecar_files(image):
        arc = relative_arc_name(sidecar, base_dir)
        if arc in seen:
            continue
        seen.add(arc)
        zf.write(sidecar, arc)

    return added
