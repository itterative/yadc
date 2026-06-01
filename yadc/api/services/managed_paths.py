"""Managed dataset path conventions — directory layout, constants, and resolution.

A "managed" dataset is one created via file upload (``source == "upload"``) and
stored under ``STATE_PATH/datasets/<name>/`` with the standard layout::

    <config_dir>/
        config.toml
        images/                 # root files
        folders/
            train/             # or other folder names
                img.jpg

This module is the single source of truth for that layout. It exposes:

- The layout constants (``MANAGED_IMAGES_PREFIX``, ``MANAGED_FOLDERS_PREFIX``)
  used as path prefixes throughout the codebase.
- Path builders (``managed_base_dir``, ``managed_images_dir``,
  ``managed_folders_dir``) so callers don't reinvent the layout.
- The path-resolver ``compute_delete_path`` used by the image-listing endpoint
  to populate :attr:`ImageInfo.delete_path`.

It has no service dependencies and is safe to import from anywhere — both
:class:`DatasetService` and :class:`ManagedDatasetsService` consume it.
"""

from pathlib import Path

# Top-level directory names inside a managed dataset's config dir.
MANAGED_IMAGES_PREFIX: str = "images"
MANAGED_FOLDERS_PREFIX: str = "folders"


def managed_base_dir(config_path: str | Path) -> Path:
    """Return the base directory of a managed dataset (parent of ``config.toml``)."""
    return Path(config_path).parent


def managed_images_dir(config_path: str | Path) -> Path:
    """Return the absolute path to the ``images/`` directory of a managed dataset."""
    return managed_base_dir(config_path) / MANAGED_IMAGES_PREFIX


def managed_folders_dir(config_path: str | Path) -> Path:
    """Return the absolute path to the ``folders/`` directory of a managed dataset."""
    return managed_base_dir(config_path) / MANAGED_FOLDERS_PREFIX


def compute_delete_path(image_path: str, config_path: str | None) -> str | None:
    """Compute the API-facing delete path for an image based on its location.

    Returns ``images/foo.jpg`` for root files, ``folders/train/foo.jpg`` for
    folder files, or ``None`` if the image is not under a managed layout
    (external/imported datasets).
    """
    if not config_path:
        return None
    p = Path(image_path)
    try:
        rel = p.relative_to(managed_images_dir(config_path))
        return f"{MANAGED_IMAGES_PREFIX}/{rel}"
    except ValueError:
        pass
    try:
        rel = p.relative_to(managed_folders_dir(config_path))
        return f"{MANAGED_FOLDERS_PREFIX}/{rel}"
    except ValueError:
        pass
    return None
