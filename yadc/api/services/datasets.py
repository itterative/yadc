"""Dataset service — filesystem scanning, SQLite indexing, and image queries.

A "dataset" is a named yadc config TOML stored in the XDG state directory
(``STATE_PATH/datasets/<name>/config.toml``). The TOML is the
source of truth — it defines API settings, prompts, and ``[[dataset]]``
entries pointing to image directories.

Two registration flows:

- **Import**: register an existing TOML path (resolving relative paths to
  absolute first).
- **Create**: write a fresh TOML with the given image paths.

Both end up as ``STATE_PATH/datasets/<name>/config.toml``.
"""

import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any

import tomlkit
from PIL import Image

from yadc.cmd.app import STATE_PATH
from yadc.core.config import Config, parse_config
from yadc.core.dataset import DatasetImage
from yadc.utils.dict_utils import toml_to_plain

from ..configuration import Configuration
from ..events import DatasetChangedEvent
from ..modules.dataset_watcher import SELF_JOB_ID, SIDECAR_EXTENSION_GLOBS, DatasetWatcherService
from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.event_dispatcher import EventDispatcher, event_handler
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service

# Image extensions we recognize (matching what PIL can open).
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".ico"})

# Base directory for all dataset state directories.
DATASETS_DIR: Path = STATE_PATH / "datasets"

# Prefixes for managed dataset directory layout (used in config paths and deletion API).
MANAGED_IMAGES_PREFIX: str = "images"
MANAGED_FOLDERS_PREFIX: str = "folders"


def _dataset_config_path(name: str) -> Path:
    """Return the config TOML path for a named dataset."""
    return DATASETS_DIR / name / "config.toml"


@dataclass
class DatasetInfo:
    """Summary of a dataset as returned by the listing API."""

    name: str
    source: str = "import"
    config_path: str | None = None
    image_count: int = 0
    has_caption: int = 0
    has_toml: int = 0
    last_scanned_t: float | None = None
    first_image_id: int | None = None


@dataclass
class ImageInfo:
    """Summary of a single image within a dataset."""

    id: int
    file_name: str
    path: str
    has_caption: bool = False
    has_toml: bool = False
    width: int = 0
    height: int = 0
    draft_names: list[str] = field(default_factory=list)
    last_modified_t: float | None = None
    delete_path: str | None = None


@dataclass
class HistoryEntry:
    """A single history snapshot for an image."""

    index: int
    caption: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImagePage:
    """A paginated page of images."""

    images: list[ImageInfo]
    next_token: str | None = None


class DatasetService(Service):
    """Scans filesystem for dataset configs, indexes images in SQLite, and serves queries.

    A "dataset" is a directory containing a ``config.toml`` (v1 or v2 format) with
    ``[[dataset]]`` entries pointing to image directories. The service scans the
    yadc config directories and any registered paths, indexes image metadata into
    the ``datasets`` and ``dataset_images`` tables, and provides paginated queries.
    """

    def __init__(
        self,
        db: DBConnectionFactory,
        watcher: DatasetWatcherService,
        configuration: Configuration,
        event_dispatcher: EventDispatcher,
        logging: LoggingFactory,
    ):
        self._db: DBConnectionFactory = db
        self._watcher: DatasetWatcherService = watcher
        self._configuration: Configuration = configuration
        self._logger: Logger = logging.get_logger(__name__)
        # Ensure state dir exists
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)

        # Configure watcher and register existing datasets
        self._watch_existing_datasets()

        # Register for DatasetChangedEvent to auto-rescan
        event_dispatcher.register_service(self)

    # --- Dataset listing ---

    def list_datasets(self) -> list[DatasetInfo]:
        """Return all indexed datasets, refreshing stale entries first."""
        self._refresh_stale_datasets()

        conn = self._db.connection()
        try:
            rows = conn.execute(
                """
                SELECT d.name, d.source, d.config_path,
                       d.image_count,
                       COALESCE(ci.has_caption, 0),
                       COALESCE(ci.has_toml, 0),
                       d.last_scanned_t,
                       (SELECT MIN(di2.id) FROM dataset_images di2 WHERE di2.dataset_id = d.id)
                FROM datasets d
                LEFT JOIN (
                    SELECT dataset_id,
                           SUM(has_caption) AS has_caption,
                           SUM(has_toml) AS has_toml
                    FROM dataset_images
                    GROUP BY dataset_id
                ) ci ON ci.dataset_id = d.id
                ORDER BY d.name
                """
            ).fetchall()

            return [
                DatasetInfo(
                    name=row[0],
                    source=row[1],
                    config_path=row[2],
                    image_count=row[3],
                    has_caption=row[4],
                    has_toml=row[5],
                    last_scanned_t=row[6],
                    first_image_id=row[7],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def get_dataset(self, name: str) -> DatasetInfo | None:
        """Get a single dataset by name, refreshing if stale."""
        self._refresh_stale_datasets()

        conn = self._db.connection()
        try:
            row = conn.execute(
                """
                SELECT d.name, d.source, d.config_path,
                       d.image_count,
                       COALESCE(ci.has_caption, 0),
                       COALESCE(ci.has_toml, 0),
                       d.last_scanned_t,
                       (SELECT MIN(di2.id) FROM dataset_images di2 WHERE di2.dataset_id = d.id)
                FROM datasets d
                LEFT JOIN (
                    SELECT dataset_id,
                           SUM(has_caption) AS has_caption,
                           SUM(has_toml) AS has_toml
                    FROM dataset_images
                    WHERE dataset_id = (SELECT id FROM datasets WHERE name = ?)
                    GROUP BY dataset_id
                ) ci ON ci.dataset_id = d.id
                WHERE d.name = ?
                """,
                (name, name),
            ).fetchone()

            if row is None:
                return None

            return DatasetInfo(
                name=row[0],
                source=row[1],
                config_path=row[2],
                image_count=row[3],
                has_caption=row[4],
                has_toml=row[5],
                last_scanned_t=row[6],
                first_image_id=row[7],
            )
        finally:
            conn.close()

    # --- Image listing (paginated) ---

    def list_images(self, dataset_name: str, *, limit: int = 50, after_id: int = 0) -> ImagePage:
        """List images for a dataset, paginated by auto-increment ID.

        Args:
            dataset_name: Name of the dataset.
            limit: Max images to return.
            after_id: Cursor — return images with id > this value.

        Returns:
            An ImagePage with images and optional next_token.
        """
        conn = self._db.connection()
        try:
            rows = conn.execute(
                """
                SELECT di.id, di.file_name, di.path, di.has_caption, di.has_toml, di.width, di.height, di.draft_names, di.last_modified_t, d.config_path
                FROM dataset_images di
                JOIN datasets d ON d.id = di.dataset_id
                WHERE d.name = ? AND di.id > ?
                ORDER BY di.id
                LIMIT ?
                """,
                (dataset_name, after_id, limit + 1),
            ).fetchall()

            images = [
                ImageInfo(
                    id=row[0],
                    file_name=row[1],
                    path=row[2],
                    has_caption=bool(row[3]),
                    has_toml=bool(row[4]),
                    width=row[5] or 0,
                    height=row[6] or 0,
                    draft_names=row[7].split(",") if row[7] else [],
                    last_modified_t=row[8],
                    delete_path=self._compute_delete_path(row[2], row[9]),
                )
                for row in rows[:limit]
            ]

            next_token = None
            if len(rows) > limit:
                next_token = str(images[-1].id)

            return ImagePage(images=images, next_token=next_token)
        finally:
            conn.close()

    def get_image(self, dataset_name: str, image_id: int) -> ImageInfo | None:
        """Get a single image by ID within a dataset."""
        conn = self._db.connection()
        try:
            row = conn.execute(
                """
                SELECT di.id, di.file_name, di.path, di.has_caption, di.has_toml, di.width, di.height, di.draft_names, di.last_modified_t, d.config_path
                FROM dataset_images di
                JOIN datasets d ON d.id = di.dataset_id
                WHERE d.name = ? AND di.id = ?
                """,
                (dataset_name, image_id),
            ).fetchone()

            if row is None:
                return None

            return ImageInfo(
                id=row[0],
                file_name=row[1],
                path=row[2],
                has_caption=bool(row[3]),
                has_toml=bool(row[4]),
                width=row[5] or 0,
                height=row[6] or 0,
                draft_names=row[7].split(",") if row[7] else [],
                last_modified_t=row[8],
                delete_path=self._compute_delete_path(row[2], row[9]),
            )
        finally:
            conn.close()

    # --- Image content access ---

    def get_image_path(self, dataset_name: str, image_id: int) -> Path | None:
        """Return the filesystem path for an image, or None if not found."""
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return None
        p = Path(info.path)
        return p if p.exists() else None

    def get_image_by_path(self, dataset_name: str, image_path: str | Path) -> ImageInfo | None:
        """Look up an image by its filesystem path within a dataset."""
        conn = self._db.connection()
        try:
            row = conn.execute(
                """
                SELECT di.id, di.file_name, di.path, di.has_caption, di.has_toml, di.width, di.height, di.draft_names, di.last_modified_t
                FROM dataset_images di
                JOIN datasets d ON d.id = di.dataset_id
                WHERE d.name = ? AND di.path = ?
                """,
                (dataset_name, str(image_path)),
            ).fetchone()

            if row is None:
                return None

            return ImageInfo(
                id=row[0],
                file_name=row[1],
                path=row[2],
                has_caption=bool(row[3]),
                has_toml=bool(row[4]),
                width=row[5] or 0,
                height=row[6] or 0,
                draft_names=row[7].split(",") if row[7] else [],
                last_modified_t=row[8],
            )
        finally:
            conn.close()

    def get_draft_names(self, dataset_name: str) -> list[str]:
        """Return sorted list of unique draft names across all images in a dataset."""
        conn = self._db.connection()
        try:
            row = conn.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,)).fetchone()
            if row is None:
                return []
            dataset_id: int = row[0]
            raw = conn.execute(
                """
                SELECT DISTINCT draft_names FROM dataset_images
                WHERE dataset_id = ? AND draft_names != ''
                """,
                (dataset_id,),
            ).fetchall()
            names: set[str] = set()
            for (csv,) in raw:
                for part in csv.split(","):
                    part = part.strip()
                    if part:
                        names.add(part)
            return sorted(names)
        finally:
            conn.close()

    def get_caption(self, dataset_name: str, image_id: int) -> dict[str, Any] | None:
        """Read the caption text and TOML extras for an image.

        Returns dict with keys: caption, extras, drafts — or None if image not found.
        """
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return None

        image_path = Path(info.path)
        if not image_path.exists():
            return None

        dataset_image = DatasetImage(path=str(image_path))
        caption = dataset_image.read_caption()

        extras_raw: str = ""
        extras: dict[str, Any] = {}
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras_raw = f.read()
                    f.seek(0)
                    extras = tomlkit.loads(extras_raw)
            except Exception:
                pass

        drafts: dict[str, str] = {}
        try:
            drafts = dataset_image.read_all_drafts()
        except Exception:
            pass

        return {"caption": caption, "extras": extras, "extras_raw": extras_raw, "drafts": drafts}

    def preview_prompt(self, dataset_name: str, image_id: int, template: str) -> dict[str, Any] | None:
        """Render the system and user prompts for an image using a Jinja2 template.

        Returns dict with keys: system_prompt, user_prompt, template_context — or None if not found.
        """
        from yadc.core.captioner import PromptRenderer

        info = self.get_image(dataset_name, image_id)
        if info is None:
            return None

        image_path = Path(info.path)
        if not image_path.exists():
            return None

        dataset_image = DatasetImage(path=str(image_path))

        # Load extras from TOML sidecar
        extras: dict[str, Any] = {}
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras = tomlkit.loads(f.read())
            except Exception:
                pass

        # Apply extras as additional fields on the DatasetImage
        if extras:
            dataset_image = DatasetImage.model_validate({"path": str(image_path), **extras})

        # Read current caption from .txt file
        dataset_image.caption = dataset_image.read_caption()

        # Load drafts
        drafts: dict[str, str] = {}
        try:
            drafts = dataset_image.read_all_drafts()
        except Exception:
            pass

        # Build template context (same as what PromptRenderer.render uses)
        template_context = dataset_image.model_dump()
        if drafts:
            template_context["drafts"] = drafts

        renderer = PromptRenderer(prompt_template=template)
        system_prompt, user_prompt = renderer.render(dataset_image, drafts=drafts or None)

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "template_context": template_context,
            "template_context_toml": tomlkit.dumps(template_context),
        }

    def get_history(self, dataset_name: str, image_id: int, limit: int = 3) -> list[HistoryEntry] | None:
        """Read history entries for an image, most recent first.

        Returns None if image not found, empty list if no history.
        """
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return None

        image_path = Path(info.path)
        if not image_path.exists():
            return None

        dataset_image = DatasetImage(path=str(image_path))
        all_entries = dataset_image.read_history()

        # Take the last `limit` entries and reverse (most recent first)
        recent = all_entries[-limit:] if len(all_entries) > limit else all_entries
        recent.reverse()

        result: list[HistoryEntry] = []
        for i, entry in enumerate(recent):
            extras = dict(entry.__pydantic_extra__ or {})
            result.append(HistoryEntry(index=len(recent) - 1 - i, caption=entry.caption, extras=extras))

        # Re-index using absolute positions from the end of the full list
        total = len(all_entries)
        for i, entry in enumerate(result):
            entry.index = total - 1 - i

        return result

    def restore_history(self, dataset_name: str, image_id: int, history_index: int, *, source: str = SELF_JOB_ID) -> bool:
        """Restore caption + extras from a history entry.

        Saves the current state to history before restoring.
        Returns True on success.
        """
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        dataset_image = DatasetImage(path=str(image_path))

        # Register expected file changes so the watcher suppresses the notification.
        self._watcher.expect_file_change(dataset_name, str(dataset_image.history_path), source=source)
        self._watcher.expect_file_change(dataset_name, str(dataset_image.caption_path), source=source)
        self._watcher.expect_file_change(dataset_name, str(dataset_image.toml_path), source=source)

        # Load current extras into the model so save_history captures them
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras = tomlkit.loads(f.read())
                if extras:
                    dataset_image = DatasetImage.model_validate({"path": str(image_path), **extras})
            except Exception:
                pass

        # Read current caption
        dataset_image.caption = dataset_image.read_caption()

        # Save current state to history before restoring
        dataset_image.save_history()

        # Read all history and find the target entry
        all_entries = dataset_image.read_history()
        if history_index < 0 or history_index >= len(all_entries):
            return False

        target = all_entries[history_index]

        # Restore caption
        dataset_image.update_caption(target.caption)

        # Restore extras
        target_extras = dict(target.__pydantic_extra__ or {})
        if target_extras:
            dataset_image.toml_path.write_text(tomlkit.dumps(target_extras))
            self._update_image_index(image_id, has_toml=True)
        else:
            # Clear extras if the history entry had none
            if dataset_image.toml_path.exists():
                dataset_image.toml_path.write_text("")
            self._update_image_index(image_id, has_toml=False)

        self._update_image_index(image_id, has_caption=True)
        return True

    def update_caption(self, dataset_name: str, image_id: int, caption: str, *, source: str = SELF_JOB_ID) -> bool:
        """Update the caption file for an image. Saves history first. Returns True on success."""
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        dataset_image = DatasetImage(path=str(image_path))

        # Register expected file changes so the watcher suppresses the notification.
        self._watcher.expect_file_change(dataset_name, str(dataset_image.history_path), source=source)
        self._watcher.expect_file_change(dataset_name, str(dataset_image.caption_path), source=source)
        self._watcher.expect_file_change(dataset_name, str(dataset_image.toml_path), source=source)

        # Load current state and save to history before overwriting
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras = tomlkit.loads(f.read())
                if extras:
                    dataset_image = DatasetImage.model_validate({"path": str(image_path), **extras})
            except Exception:
                pass
        dataset_image.caption = dataset_image.read_caption()
        dataset_image.save_history()

        # Now apply the update
        dataset_image = DatasetImage(path=str(image_path))
        dataset_image.update_caption(caption)

        # Update the index
        self._update_image_index(image_id, has_caption=True)
        return True

    def update_extras(self, dataset_name: str, image_id: int, extras_raw: str, *, source: str = SELF_JOB_ID) -> bool:
        """Update the TOML extras sidecar for an image. Saves history first. Returns True on success."""
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        # Validate TOML before writing
        try:
            tomlkit.loads(extras_raw)
        except Exception as e:
            raise ValueError(f"Invalid TOML: {e}") from e

        dataset_image = DatasetImage(path=str(image_path))

        # Register expected file changes so the watcher suppresses the notification.
        self._watcher.expect_file_change(dataset_name, str(dataset_image.history_path), source=source)
        self._watcher.expect_file_change(dataset_name, str(dataset_image.toml_path), source=source)

        # Load current state and save to history before overwriting
        current_extras: dict[str, Any] = {}
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    current_extras = tomlkit.loads(f.read())
                if current_extras:
                    dataset_image = DatasetImage.model_validate({"path": str(image_path), **current_extras})
            except Exception:
                pass
        dataset_image.caption = dataset_image.read_caption()
        dataset_image.save_history()

        # Now write the new extras
        dataset_image.toml_path.write_text(extras_raw)

        self._update_image_index(image_id, has_toml=bool(extras_raw.strip()))
        return True

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
        info = self.get_dataset(name)
        if info is None:
            raise ValueError(f"Dataset '{name}' not found")
        if info.source != "upload":
            raise ValueError(f"Dataset '{name}' is not a managed dataset")
        if not info.config_path:
            raise ValueError(f"Dataset '{name}' has no config path")

        base_dir = Path(info.config_path).parent
        images_dir = base_dir / MANAGED_IMAGES_PREFIX
        folders_dir = base_dir / MANAGED_FOLDERS_PREFIX

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
        self.rescan_dataset(name)
        return deleted, warnings

    def _compute_delete_path(self, image_path: str, config_path: str | None) -> str | None:
        """Compute the API-facing delete path for an image based on its location.

        Returns ``images/foo.jpg`` for root files, ``folders/train/foo.jpg`` for
        folder files, or ``None`` if the image is not under a managed layout.
        """
        if not config_path:
            return None
        config_file = Path(config_path)
        base_dir = config_file.parent
        images_dir = base_dir / MANAGED_IMAGES_PREFIX
        folders_dir = base_dir / MANAGED_FOLDERS_PREFIX
        p = Path(image_path)
        try:
            rel = p.relative_to(images_dir)
            return f"{MANAGED_IMAGES_PREFIX}/{rel}"
        except ValueError:
            pass
        try:
            rel = p.relative_to(folders_dir)
            return f"{MANAGED_FOLDERS_PREFIX}/{rel}"
        except ValueError:
            pass
        return None

    def _remove_folder_dataset_entries(self, config_path_str: str, folder_names: list[str]) -> None:
        """Remove ``[[dataset]]`` entries for deleted folders from config TOML."""
        if not folder_names:
            return
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

    def list_folders(self, name: str) -> list[dict[str, Any]]:
        """List folders for a managed dataset with image counts.

        Returns a list of dicts with keys: name, path, image_count.
        The root ``images/`` folder is always listed first (if it exists).

        ``path`` is the value to pass to :meth:`delete_items` to remove
        the folder (e.g. ``"train"`` deletes ``folders/train/``).
        """
        info = self.get_dataset(name)
        if info is None:
            raise ValueError(f"Dataset '{name}' not found")
        if info.source != "upload":
            raise ValueError(f"Dataset '{name}' is not a managed dataset")
        if not info.config_path:
            raise ValueError(f"Dataset '{name}' has no config path")

        base_dir = Path(info.config_path).parent
        images_dir = base_dir / "images"
        folders_dir = base_dir / "folders"

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

    # --- Dataset registration ---

    def import_dataset(self, name: str, toml_path: str) -> DatasetInfo:
        """Import an existing TOML config as a named dataset.

        Resolves any relative paths in ``[[dataset]]`` entries to absolute
        and writes them back to the original TOML file. The dataset is
        registered with the original path (no local copy is kept).
        """
        source = Path(toml_path).resolve()
        if not source.is_file():
            raise ValueError(f"Not a file: {source}")

        # Load and resolve relative paths
        raw = self._load_raw_config(source)
        raw = self._resolve_relative_paths(raw, source.parent)

        # Write resolved paths back to the original file
        with open(source, "w") as f:
            tomlkit.dump(raw, f)

        return self.register(name, str(source), source="import")

    def create_dataset(self, name: str, image_paths: list[str]) -> DatasetInfo:
        """Create a new dataset with the given image directories.

        Writes a minimal TOML to ``STATE_PATH/datasets/<name>/config.toml``
        with one ``[[dataset]]`` entry per path.
        """
        if not image_paths:
            raise ValueError("At least one image path is required")

        resolved_paths: list[str] = []
        for p in image_paths:
            rp = Path(p).resolve()
            if not rp.is_dir():
                raise ValueError(f"Not a directory: {rp}")
            resolved_paths.append(str(rp))

        # Build a minimal v2 TOML
        raw: dict[str, Any] = {"dataset": []}
        for ip in resolved_paths:
            raw["dataset"].append({"path": ip})

        dest_dir = DATASETS_DIR / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _dataset_config_path(name)
        with open(dest, "w") as f:
            tomlkit.dump(raw, f)

        return self.register(name, str(dest), source="create")

    def rescan_dataset(self, name: str) -> bool:
        """Force a rescan of a dataset by name. Returns True if dataset was found."""
        conn = self._db.connection()
        try:
            row = conn.execute("SELECT id, config_path FROM datasets WHERE name = ?", (name,)).fetchone()
            if row is None:
                return False

            dataset_id, config_path = row[0], row[1]
            self._scan_dataset(conn, dataset_id, config_path)
            conn.commit()

            # Re-register filesystem watcher with current paths (picks up path changes)
            if config_path:
                config_path_obj = Path(config_path)
                config = self._load_config(config_path_obj)
                paths = self._get_dataset_paths(config, config_path_obj)
                self._watcher.watch_dataset(name, paths)

            return True
        finally:
            conn.close()

    def unregister_dataset(self, name: str) -> bool:
        """Remove a dataset, its images from the index, and its state dir."""

        self._watcher.unwatch_dataset(name)

        conn = self._db.connection()
        try:
            cursor = conn.execute("DELETE FROM datasets WHERE name = ?", (name,))
            conn.commit()
            found = cursor.rowcount > 0
        finally:
            conn.close()

        if found:
            state_dir = DATASETS_DIR / name
            if state_dir.is_dir():
                shutil.rmtree(state_dir, ignore_errors=True)

        return found

    def register(self, name: str, config_path: str, *, source: str = "import") -> DatasetInfo:
        """Upsert a dataset record and scan its images."""
        config = self._load_config(Path(config_path))

        conn = self._db.connection()
        try:
            conn.execute(
                """
                INSERT INTO datasets (name, source, config_path, last_scanned_t, updated_t)
                VALUES (?, ?, ?, unixepoch(), unixepoch())
                ON CONFLICT(name) DO UPDATE SET
                    config_path = excluded.config_path,
                    source = excluded.source,
                    updated_t = unixepoch()
                """,
                (name, source, config_path),
            )

            row = conn.execute("SELECT id FROM datasets WHERE name = ?", (name,)).fetchone()
            assert row is not None
            dataset_id = row[0]

            self._scan_dataset(conn, dataset_id, config_path, config=config)
            conn.commit()
        finally:
            conn.close()

        # Start watching the dataset's directories
        paths = self._get_dataset_paths(config, Path(config_path))
        if paths:
            self._watcher.watch_dataset(name, paths)

        result = self.get_dataset(name)
        assert result is not None
        return result

    # --- Private helpers ---

    def _load_raw_config(self, config_path: Path) -> dict[str, Any]:
        """Load a TOML config as a raw dict."""
        with open(config_path) as f:
            return tomlkit.load(f)

    def _resolve_relative_paths(self, raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
        """Resolve relative dataset paths in a raw config dict to absolute paths."""
        # Handle v2 [[dataset]]
        dataset_entries = raw.get("dataset")
        if isinstance(dataset_entries, list):
            for entry in dataset_entries:
                if isinstance(entry, dict) and "path" in entry:
                    p = Path(entry["path"])
                    if not p.is_absolute():
                        entry["path"] = str((base_dir / p).resolve())

        # Handle v1 [dataset] paths
        dataset_v1 = raw.get("dataset")
        if isinstance(dataset_v1, dict) and "paths" in dataset_v1:
            resolved = []
            for p in dataset_v1["paths"]:
                pp = Path(p)
                if not pp.is_absolute():
                    resolved.append(str((base_dir / pp).resolve()))
                else:
                    resolved.append(p)
            dataset_v1["paths"] = resolved

        return raw

    def _load_config(self, config_path: Path) -> Config | None:
        """Load and parse a dataset config file with relaxed validation.

        Uses ``strict=False`` so webui TOMLs (missing api_url/model/template)
        parse cleanly. Those fields are validated when creating the captioner.
        """
        try:
            with open(config_path) as f:
                raw = tomlkit.load(f)
            return parse_config(toml_to_plain(raw), strict=False)
        except Exception as e:
            self._logger.warning("Failed to parse config at %s: %s", config_path, e)
            return None

    def _scan_dataset(
        self,
        conn: sqlite3.Connection,
        dataset_id: int,
        config_path: str | None,
        config: Config | None = None,
    ) -> None:
        """Scan a dataset's config and update the image index."""
        if not config_path:
            return

        config_file = Path(config_path)
        if not config_file.exists():
            return

        # Load config if not provided
        if config is None:
            config = self._load_config(config_file)

        # Determine image directories from config (paths are already absolute)
        image_dirs: list[Path] = []
        extras_per_dir: dict[str, dict[str, object]] = {}

        if config is None:
            return

        for entry in config.dataset:
            if entry.path:
                p = Path(entry.path)
                if not p.is_absolute():
                    p = config_file.parent / p
                if p.is_dir():
                    image_dirs.append(p.resolve())
                    extras_per_dir[str(p.resolve())] = entry.extras

        if not image_dirs:
            return

        # Collect current files from disk
        disk_images: dict[str, dict[str, Any]] = {}
        for img_dir in image_dirs:
            for child in sorted(img_dir.iterdir()):
                if not child.is_file():
                    continue
                if child.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                caption_path = child.with_suffix(".txt")
                toml_path = child.with_suffix(".toml")

                # Find drafts
                draft_names: list[str] = []
                stem = child.stem
                for f in child.parent.iterdir():
                    if f.name.startswith(stem + ".") and f.name.endswith(".draft~"):
                        draft_name = f.name[len(stem) + 1 : -len(".draft~")]
                        if draft_name:
                            draft_names.append(draft_name)

                try:
                    mod_time = child.stat().st_mtime
                except OSError:
                    mod_time = None

                # Read image dimensions
                img_width = 0
                img_height = 0
                try:
                    with Image.open(child) as img:
                        img_width, img_height = img.size
                except Exception:
                    pass

                disk_images[str(child)] = {
                    "file_name": child.name,
                    "has_caption": caption_path.exists(),
                    "has_toml": toml_path.exists(),
                    "width": img_width,
                    "height": img_height,
                    "draft_names": ",".join(sorted(draft_names)),
                    "last_modified_t": mod_time,
                }

        # Get existing indexed images
        existing_rows = conn.execute(
            "SELECT id, path FROM dataset_images WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchall()
        existing_by_path: dict[str, int] = {row[1]: row[0] for row in existing_rows}

        # Upsert disk images
        for img_path, meta in disk_images.items():
            if img_path in existing_by_path:
                conn.execute(
                    """
                    UPDATE dataset_images
                    SET file_name = ?, has_caption = ?, has_toml = ?, width = ?, height = ?, draft_names = ?, last_modified_t = ?
                    WHERE id = ?
                    """,
                    (
                        meta["file_name"],
                        int(meta["has_caption"]),
                        int(meta["has_toml"]),
                        meta["width"],
                        meta["height"],
                        meta["draft_names"],
                        meta["last_modified_t"],
                        existing_by_path[img_path],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO dataset_images (dataset_id, path, file_name, has_caption, has_toml, width, height, draft_names, last_modified_t)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        img_path,
                        meta["file_name"],
                        int(meta["has_caption"]),
                        int(meta["has_toml"]),
                        meta["width"],
                        meta["height"],
                        meta["draft_names"],
                        meta["last_modified_t"],
                    ),
                )

        # Remove images that no longer exist on disk
        for img_path, img_id in existing_by_path.items():
            if img_path not in disk_images:
                conn.execute("DELETE FROM dataset_images WHERE id = ?", (img_id,))

        # Update dataset metadata
        conn.execute(
            """
            UPDATE datasets
            SET image_count = ?, last_scanned_t = unixepoch(), updated_t = unixepoch()
            WHERE id = ?
            """,
            (len(disk_images), dataset_id),
        )

    def _update_image_index(self, image_id: int, *, has_caption: bool | None = None, has_toml: bool | None = None) -> None:
        """Update specific fields on an indexed image."""
        conn = self._db.connection()
        try:
            sets: list[str] = []
            params: list[Any] = []
            if has_caption is not None:
                sets.append("has_caption = ?")
                params.append(int(has_caption))
            if has_toml is not None:
                sets.append("has_toml = ?")
                params.append(int(has_toml))
            if not sets:
                return
            params.append(image_id)
            conn.execute(f"UPDATE dataset_images SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()
        finally:
            conn.close()

    def refresh_image_index(self, dataset_name: str, image_id: int) -> None:
        """Re-read a single image's disk state and update its index row."""
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return

        image_path = Path(info.path)
        if not image_path.exists():
            return

        caption_path = image_path.with_suffix(".txt")
        toml_path = image_path.with_suffix(".toml")

        # Collect draft names
        draft_names: list[str] = []
        stem = image_path.stem
        for f in image_path.parent.iterdir():
            if f.name.startswith(stem + ".") and f.name.endswith(".draft~"):
                draft_name = f.name[len(stem) + 1 : -len(".draft~")]
                if draft_name:
                    draft_names.append(draft_name)

        conn = self._db.connection()
        try:
            conn.execute(
                "UPDATE dataset_images SET has_caption = ?, has_toml = ?, draft_names = ? WHERE id = ?",
                (int(caption_path.exists()), int(toml_path.exists()), ",".join(sorted(draft_names)), image_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _refresh_stale_datasets(self, max_age_seconds: float = 60.0) -> None:
        """Rescan datasets that haven't been scanned recently."""
        conn = self._db.connection()
        try:
            cutoff = time.time() - max_age_seconds
            stale = conn.execute(
                "SELECT id, name, config_path FROM datasets WHERE last_scanned_t IS NULL OR last_scanned_t < ?",
                (cutoff,),
            ).fetchall()

            for dataset_id, name, config_path in stale:
                try:
                    self._scan_dataset(conn, dataset_id, config_path)
                    conn.commit()

                    # Re-register filesystem watcher so path changes are tracked
                    if config_path:
                        config_path_obj = Path(config_path)
                        config = self._load_config(config_path_obj)
                        paths = self._get_dataset_paths(config, config_path_obj)
                        self._watcher.watch_dataset(name, paths)
                except Exception as e:
                    self._logger.warning("Failed to refresh dataset id=%d: %s", dataset_id, e)
        finally:
            conn.close()

    def _get_dataset_paths(self, config: Config | None, config_path: Path | None = None) -> list[str]:
        """Extract image directory paths from a parsed config.

        Relative paths are resolved against ``config_path.parent`` if provided.
        """
        if config is None:
            return []
        paths: list[str] = []
        for entry in config.dataset:
            if entry.path:
                p = Path(entry.path)
                if not p.is_absolute() and config_path is not None:
                    p = config_path.parent / p
                p = p.resolve()
                if p.is_dir():
                    paths.append(str(p))
        return paths

    def _watch_existing_datasets(self) -> None:
        """Watch all existing datasets' directories. Called during startup."""
        conn = self._db.connection()
        try:
            rows = conn.execute("SELECT name, config_path FROM datasets").fetchall()
        finally:
            conn.close()

        for name, config_path in rows:
            if not config_path:
                continue
            config_path_obj = Path(config_path)
            config = self._load_config(config_path_obj)
            paths = self._get_dataset_paths(config, config_path_obj)
            if paths:
                self._watcher.watch_dataset(name, paths)
                self._logger.debug("Watching existing dataset. [dataset=%s, paths=%d]", name, len(paths))

    @event_handler(DatasetChangedEvent)
    def _on_dataset_changed(self, event: DatasetChangedEvent) -> None:
        """Auto-rescan when the watcher detects filesystem changes.

        Skips the rescan when the change was self-originated (webui edit or
        captioning job) — the API endpoints and per-image ``refresh_image_index``
        calls already keep the DB up to date for those.  Only truly external
        changes (new files, deletions by other processes) trigger a full rescan.
        """
        if event.job_id:
            self._logger.debug(
                "Skipping auto-rescan for self-originated change. [dataset=%s, job_id=%s]",
                event.dataset_name,
                event.job_id,
            )
            return

        self._logger.debug("Auto-rescanning dataset due to filesystem change. [dataset=%s]", event.dataset_name)
        self.rescan_dataset(event.dataset_name)
