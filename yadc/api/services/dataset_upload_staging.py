"""Staging helpers for dataset append uploads.

Staging separates the *upload* (write files to a temporary directory) from
the *commit* (apply user-chosen conflict resolutions to the live dataset).
This module owns:

- Periodic cleanup of stale staging directories.
- Building grouped views of staged files (image + sidecars per stem).
- Detecting conflicts between staged and live files.
- Filtering orphan sidecars (sidecars with no matching image stem).

Functions are stateless and operate on ``pathlib.Path`` arguments. They do
not depend on :class:`DatasetUploadService`.
"""

import shutil
import time
from collections import defaultdict
from logging import Logger
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from yadc.api.services.datasets import IMAGE_EXTENSIONS
from yadc.api.services.managed_paths import MANAGED_FOLDERS_PREFIX, MANAGED_IMAGES_PREFIX

# Name of the staging directory under each dataset's base dir.
STAGING_DIR_NAME: str = ".staging"

# Stale staging directories older than this are removed by the cleanup job.
STAGING_MAX_AGE_SECONDS: int = 24 * 3600


def cleanup_staging_dirs(datasets_dir: Path, logger: Logger) -> None:
    """Remove staging subdirectories older than ``STAGING_MAX_AGE_SECONDS``.

    Walks ``datasets_dir/<name>/.staging/<uuid>`` and deletes any subdirs
    whose mtime is older than the threshold. Also removes empty
    ``.staging`` directories afterwards.
    """
    now = time.time()
    if not datasets_dir.exists():
        return
    for base_dir in datasets_dir.iterdir():
        if not base_dir.is_dir():
            continue
        staging_dir = base_dir / STAGING_DIR_NAME
        if not staging_dir.exists():
            continue
        for subdir in staging_dir.iterdir():
            try:
                mtime = subdir.stat().st_mtime
                if now - mtime > STAGING_MAX_AGE_SECONDS:
                    shutil.rmtree(subdir, ignore_errors=True)
                    logger.debug("Cleaned up stale staging dir: %s", subdir)
            except Exception:
                pass
        # Remove empty .staging dirs
        try:
            if not any(staging_dir.iterdir()):
                staging_dir.rmdir()
        except Exception:
            pass


def find_live_image_stems(images_dir: Path, folders_dir: Path, image_extensions: frozenset[str]) -> set[tuple[str, str]]:
    """Walk the live dataset directories and collect ``(dir_key, stem)`` tuples.

    ``dir_key`` is ``""`` for root files and the folder name for files inside
    ``folders/<name>/``. Only files with extensions in ``image_extensions``
    are included.
    """
    stems: set[tuple[str, str]] = set()
    if images_dir.exists():
        for f in images_dir.iterdir():
            if f.is_file() and f.suffix.lower() in image_extensions:
                stems.add(("", f.stem))
    if folders_dir.exists():
        for folder_path in folders_dir.iterdir():
            if not folder_path.is_dir():
                continue
            for f in folder_path.iterdir():
                if f.is_file() and f.suffix.lower() in image_extensions:
                    stems.add((folder_path.name, f.stem))
    return stems


def build_staged_groups(
    staging_base: Path,
    image_extensions: frozenset[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Group staged files by ``(dir_key, stem)``.

    Each group is ``{"image": Path | None, "sidecars": list[Path]}``.

    - Files directly in ``staging_base/images/`` use ``dir_key=""``.
    - Files in ``staging_base/folders/<name>/`` use ``dir_key=<name>``.

    Used both during conflict detection (append) and during commit.
    """
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"image": None, "sidecars": []})
    staging_images = staging_base / MANAGED_IMAGES_PREFIX
    staging_folders = staging_base / MANAGED_FOLDERS_PREFIX

    if staging_images.exists():
        for src_path in staging_images.iterdir():
            if src_path.is_file():
                ext = src_path.suffix.lower()
                if ext in image_extensions:
                    groups[("", src_path.stem)]["image"] = src_path
                else:
                    groups[("", src_path.stem)]["sidecars"].append(src_path)

    if staging_folders.exists():
        for folder_path in staging_folders.iterdir():
            if not folder_path.is_dir():
                continue
            for src_path in folder_path.iterdir():
                if src_path.is_file():
                    ext = src_path.suffix.lower()
                    if ext in image_extensions:
                        groups[(folder_path.name, src_path.stem)]["image"] = src_path
                    else:
                        groups[(folder_path.name, src_path.stem)]["sidecars"].append(src_path)

    return groups


def detect_conflicts(
    staged_groups: dict[tuple[str, str], dict[str, Any]],
    staging_base: Path,
    images_dir: Path,
    folders_dir: Path,
) -> list[dict[str, Any]]:
    """Find conflicts between staged and live files.

    For each group, returns a single conflict entry per group that has any
    conflicting file (image or sidecar). The entry includes the
    representative file's name, existing size, and new size.
    """
    conflicts: list[dict[str, Any]] = []
    for (_dir_key, _stem), group in staged_groups.items():
        conflict_live_path: Path | None = None
        conflict_staged_path: Path | None = None

        if group["image"] is not None:
            pure = _staged_path_to_relative(group["image"], staging_base)
            live_path = _live_path_for(pure, images_dir, folders_dir)
            if live_path.exists():
                conflict_live_path = live_path
                conflict_staged_path = group["image"]
        else:
            # Sidecar-only group — check each sidecar for a live match
            for sidecar_path in group["sidecars"]:
                pure = _staged_path_to_relative(sidecar_path, staging_base)
                live_path = _live_path_for(pure, images_dir, folders_dir)
                if live_path.exists():
                    conflict_live_path = live_path
                    conflict_staged_path = sidecar_path
                    break

        if conflict_live_path is not None and conflict_staged_path is not None:
            # Representative file: image if present, else first sidecar
            rep_path = group["image"] if group["image"] is not None else group["sidecars"][0]
            rep_pure = _staged_path_to_relative(rep_path, staging_base)
            conflicts.append(
                {
                    "file": str(rep_pure),
                    "existing_size": conflict_live_path.stat().st_size,
                    "new_size": conflict_staged_path.stat().st_size,
                }
            )
    return conflicts


def filter_orphan_sidecars(
    files_to_write: list[tuple[str, BinaryIO, PurePosixPath, Path]],
    valid_stems: set[tuple[str, str]],
    warnings: list[str],
    new_folder_names: set[str] | None = None,
) -> list[tuple[str, BinaryIO, PurePosixPath, Path]]:
    """Drop sidecars whose image stem is not in ``valid_stems``.

    A sidecar is "orphan" if no image in the upload (or, for append, in the
    live dataset) matches its stem. ``.draft~`` files use a relaxed match:
    any image stem that is a prefix of the draft's filename counts.

    For append flows, pass ``new_folder_names`` to collect any new folder
    names encountered (used to update ``[[dataset]]`` entries on commit).
    For create flows, omit it.
    """
    final_files: list[tuple[str, BinaryIO, PurePosixPath, Path]] = []
    for filename, stream, pure, dest_path in files_to_write:
        ext = pure.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            final_files.append((filename, stream, pure, dest_path))
            if new_folder_names is not None and len(pure.parts) > 1:
                new_folder_names.add(pure.parts[0])
            continue

        dir_key = pure.parts[0] if len(pure.parts) > 1 else ""

        if ext == ".draft~":
            # Draft format: IMAGE_STEM.DRAFT_NAME.draft~ — any image stem that
            # is a prefix of the draft's filename counts as a match.
            matched = any(
                known_dir == dir_key and pure.name.startswith(known_stem + ".") and pure.name.endswith(".draft~") for known_dir, known_stem in valid_stems
            )
            if not matched:
                warnings.append(f"Skipped orphan sidecar (no matching image): {filename}")
                continue
        elif (dir_key, pure.stem) not in valid_stems:
            warnings.append(f"Skipped orphan sidecar (no matching image): {filename}")
            continue

        final_files.append((filename, stream, pure, dest_path))
        if new_folder_names is not None and len(pure.parts) > 1:
            new_folder_names.add(pure.parts[0])
    return final_files


def _staged_path_to_relative(staged_path: Path, staging_base: Path) -> PurePosixPath:
    """Convert a staged absolute path to a relative ``images/<file>`` or ``folders/<name>/<file>`` path.

    Strips the ``images/`` or ``folders/`` prefix from the relative path.
    """
    rel = staged_path.relative_to(staging_base).as_posix()
    images_prefix = f"{MANAGED_IMAGES_PREFIX}/"
    folders_prefix = f"{MANAGED_FOLDERS_PREFIX}/"
    if rel.startswith(images_prefix):
        return PurePosixPath(rel[len(images_prefix) :])
    if rel.startswith(folders_prefix):
        return PurePosixPath(rel[len(folders_prefix) :])
    return PurePosixPath(rel)


def _live_path_for(rel_pure: PurePosixPath, images_dir: Path, folders_dir: Path) -> Path:
    """Resolve a staged-relative path to its corresponding live path."""
    if len(rel_pure.parts) == 1:
        return images_dir / rel_pure.name
    return folders_dir / rel_pure.parts[0] / rel_pure.name
