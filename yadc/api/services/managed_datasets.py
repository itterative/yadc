"""Managed dataset operations — file/folder deletion and folder listing.

A "managed" dataset is one created via file upload (``source == "upload"``) and
stored under ``STATE_PATH/datasets/<name>/`` with the standard ``images/`` +
``folders/`` layout. This service owns all operations specific to that layout,
while :class:`DatasetService` handles generic dataset scanning and indexing.

The layout convention itself (path prefixes, path builders, the
``compute_delete_path`` resolver) lives in :mod:`.managed_paths` — this
service is a consumer, not the source of truth.

Concrete responsibilities:

- :meth:`ManagedDatasetsService.delete_items` — delete files (with sidecars)
  and folders from a managed dataset, updating ``[[dataset]]`` config entries.
- :meth:`ManagedDatasetsService.list_folders` — list ``images/`` and ``folders/*``
  with image counts and a ``can_delete`` flag.
"""

from __future__ import annotations

import shutil
from logging import Logger
from pathlib import Path
from typing import Any

from ..modules.dataset_watcher import SELF_JOB_ID, SIDECAR_EXTENSION_GLOBS, DatasetWatcherService
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .datasets import DatasetService
from .managed_paths import (
    MANAGED_FOLDERS_PREFIX,
    MANAGED_IMAGES_PREFIX,
    managed_folders_dir,
    managed_images_dir,
)


class ManagedDatasetsService(Service):
    """Operations specific to managed (upload-sourced) datasets.

    Depends on :class:`DatasetService` for dataset lookup and rescan
    (which is the source of truth for the index), and on
    :class:`DatasetWatcherService` to suppress expected-change events
    for the files we delete ourselves.
    """

    def __init__(
        self,
        datasets: DatasetService,
        watcher: DatasetWatcherService,
        logging: LoggingFactory,
    ):
        self._datasets: DatasetService = datasets
        self._watcher: DatasetWatcherService = watcher
        self._logger: Logger = logging.get_logger(__name__)

    # --- Public API ---

    def list_folders(self, name: str) -> list[dict[str, Any]]:
        """List folders for a managed dataset with image counts.

        Returns a list of dicts with keys: ``name``, ``path``, ``image_count``,
        ``can_delete``. The root ``images/`` folder is always listed first
        (if it exists) with ``can_delete=False``.

        ``path`` is the value to pass to :meth:`delete_items` to remove
        the folder (e.g. ``"folders/train"`` deletes ``folders/train/``).
        """
        info = self._datasets.get_dataset(name)
        if info is None:
            raise ValueError(f"Dataset '{name}' not found")
        if info.source != "upload":
            raise ValueError(f"Dataset '{name}' is not a managed dataset")
        if not info.config_path:
            raise ValueError(f"Dataset '{name}' has no config path")

        images_dir = managed_images_dir(info.config_path)
        folders_dir = managed_folders_dir(info.config_path)

        from .datasets import IMAGE_EXTENSIONS  # local import to avoid circular at module load

        result: list[dict[str, Any]] = []

        def _count_images(dir_path: Path) -> int:
            if not dir_path.exists():
                return 0
            return sum(1 for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)

        if images_dir.exists() and images_dir.is_dir():
            result.append(
                {
                    "name": MANAGED_IMAGES_PREFIX,
                    "path": MANAGED_IMAGES_PREFIX,
                    "image_count": _count_images(images_dir),
                    "can_delete": False,
                }
            )

        if folders_dir.exists() and folders_dir.is_dir():
            for folder_path in sorted(folders_dir.iterdir()):
                if not folder_path.is_dir():
                    continue
                result.append(
                    {
                        "name": folder_path.name,
                        "path": f"{MANAGED_FOLDERS_PREFIX}/{folder_path.name}",
                        "image_count": _count_images(folder_path),
                        "can_delete": True,
                    }
                )

        return result

    def delete_items(self, name: str, paths: list[str], *, source: str = SELF_JOB_ID) -> tuple[list[str], list[str]]:
        """Delete files and/or folders from a managed dataset.

        Only works for datasets with ``source == "upload"``. For each
        path in ``paths``:
        - If it resolves to a file in ``images/`` or ``folders/``,
          the file and all its sidecars are deleted.
        - If it resolves to a directory in ``folders/``, the entire
          directory is removed recursively.

        Returns ``(deleted, warnings)`` where ``deleted`` is the list of
        paths that were removed and ``warnings`` contains messages for
        paths that could not be found.
        """
        info = self._datasets.get_dataset(name)
        if info is None:
            raise ValueError(f"Dataset '{name}' not found")
        if info.source != "upload":
            raise ValueError(f"Dataset '{name}' is not a managed dataset")
        if not info.config_path:
            raise ValueError(f"Dataset '{name}' has no config path")

        images_dir = managed_images_dir(info.config_path)
        folders_dir = managed_folders_dir(info.config_path)
        deleted: list[str] = []
        deleted_folder_names: list[str] = []
        warnings: list[str] = []

        images_prefix = f"{MANAGED_IMAGES_PREFIX}/"
        folders_prefix = f"{MANAGED_FOLDERS_PREFIX}/"

        for rel_path in paths:
            if not rel_path:
                warnings.append("Empty path")
                continue

            rel = rel_path.strip("/")
            found = False

            # Explicit prefixed paths (preferred)
            if rel.startswith(images_prefix):
                file_path = images_dir / rel[len(images_prefix) :]
                if file_path.exists() and file_path.is_file():
                    self._delete_file_with_sidecars(file_path, name, source=source)
                    deleted.append(rel_path)
                    found = True
                else:
                    warnings.append(f"Not found: {rel_path}")
                    continue

            elif rel.startswith(folders_prefix):
                target = folders_dir / rel[len(folders_prefix) :]
                if target.exists() and target.is_file():
                    self._delete_file_with_sidecars(target, name, source=source)
                    deleted.append(rel_path)
                    found = True
                elif target.exists() and target.is_dir():
                    # Register a pattern for everything inside the folder so
                    # the watcher treats the bulk deletion as expected.
                    self._watcher.expect_pattern_change(name, str(target / "*"), source=source)
                    shutil.rmtree(target)
                    deleted.append(rel_path)
                    deleted_folder_names.append(rel[len(folders_prefix) :])
                    found = True
                else:
                    warnings.append(f"Not found: {rel_path}")
                    continue

            elif rel == MANAGED_IMAGES_PREFIX:
                warnings.append(f"Cannot delete root images folder: {rel_path}")
                continue

            # Bare paths (backward compat — individual images only)
            if not found:
                file_path = images_dir / rel
                if file_path.exists() and file_path.is_file():
                    self._delete_file_with_sidecars(file_path, name, source=source)
                    deleted.append(rel_path)
                    found = True

            if not found:
                file_path = folders_dir / rel
                if file_path.exists() and file_path.is_file():
                    self._delete_file_with_sidecars(file_path, name, source=source)
                    deleted.append(rel_path)
                    found = True

            if not found:
                folder_path = folders_dir / rel
                if folder_path.exists() and folder_path.is_dir():
                    self._watcher.expect_pattern_change(name, str(folder_path / "*"), source=source)
                    shutil.rmtree(folder_path)
                    deleted.append(rel_path)
                    deleted_folder_names.append(rel)
                    found = True

            if not found:
                warnings.append(f"Not found: {rel_path}")

        if deleted_folder_names:
            self._remove_folder_dataset_entries(info.config_path, deleted_folder_names)

        # Rescan so the SQLite index reflects the deletions
        self._datasets.rescan_dataset(name)
        return deleted, warnings

    # --- Private helpers ---

    def _delete_file_with_sidecars(self, file_path: Path, dataset_name: str = "", *, source: str = SELF_JOB_ID) -> None:
        """Delete an image file and all known sidecars in the same directory."""
        parent = file_path.parent
        stem = file_path.stem

        # Register expected file changes before deletion so the watcher
        # doesn't dispatch unexpected-change events for our own deletes.
        if dataset_name:
            self._watcher.expect_file_change(dataset_name, str(file_path), source=source)
            for suffix in SIDECAR_EXTENSION_GLOBS:
                self._watcher.expect_pattern_change(dataset_name, str(parent / f"{stem}{suffix}"), source=source)

        # Delete the image itself
        file_path.unlink(missing_ok=True)

        # Delete sidecars: .txt, .toml, .history~, and .*.draft~
        for sibling in parent.iterdir():
            if sibling.name == f"{stem}.txt":
                sibling.unlink(missing_ok=True)
            elif sibling.name == f"{stem}.toml":
                sibling.unlink(missing_ok=True)
            elif sibling.name == f"{stem}.history~":
                sibling.unlink(missing_ok=True)
            elif sibling.name.startswith(f"{stem}.") and sibling.name.endswith(".draft~"):
                sibling.unlink(missing_ok=True)

    def _remove_folder_dataset_entries(self, config_path_str: str, folder_names: list[str]) -> None:
        """Remove ``[[dataset]]`` entries for deleted folders from config TOML."""
        if not folder_names:
            return
        import tomlkit

        config_path = Path(config_path_str)
        if not config_path.exists():
            return

        with open(config_path) as f:
            doc = tomlkit.parse(f.read())

        entries = doc.get("dataset")
        if not isinstance(entries, list):
            return

        targets = {f"{MANAGED_FOLDERS_PREFIX}/{fn}" for fn in folder_names}
        new_entries = [e for e in entries if not (isinstance(e, dict) and e.get("path") in targets)]

        if len(new_entries) != len(entries):
            doc["dataset"] = new_entries
            with open(config_path, "w") as f:
                f.write(tomlkit.dumps(doc))
