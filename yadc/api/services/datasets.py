"""Dataset service — filesystem scanning, SQLite indexing, and image queries."""

import sqlite3
import time
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any

import toml

from yadc.core.config import Config, parse_config
from yadc.core.dataset import DatasetImage

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service

# Image extensions we recognize (matching what PIL can open).
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".ico"})


@dataclass
class DatasetInfo:
    """Summary of a dataset as returned by the listing API."""

    name: str
    path: str
    config_path: str | None = None
    image_count: int = 0
    has_caption: int = 0
    has_toml: int = 0
    last_scanned_t: float | None = None


@dataclass
class ImageInfo:
    """Summary of a single image within a dataset."""

    id: int
    file_name: str
    path: str
    has_caption: bool = False
    has_toml: bool = False
    draft_names: list[str] = field(default_factory=list)
    last_modified_t: float | None = None


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

    def __init__(self, db: DBConnectionFactory, logging: LoggingFactory):
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)

    # --- Dataset listing ---

    def list_datasets(self) -> list[DatasetInfo]:
        """Return all indexed datasets, refreshing stale entries first."""
        self._refresh_stale_datasets()

        conn = self._db.connection()
        try:
            rows = conn.execute(
                """
                SELECT d.name, d.path, d.config_path,
                       d.image_count,
                       COALESCE(ci.has_caption, 0),
                       COALESCE(ci.has_toml, 0),
                       d.last_scanned_t
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
                    path=row[1],
                    config_path=row[2],
                    image_count=row[3],
                    has_caption=row[4],
                    has_toml=row[5],
                    last_scanned_t=row[6],
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
                SELECT d.name, d.path, d.config_path,
                       d.image_count,
                       COALESCE(ci.has_caption, 0),
                       COALESCE(ci.has_toml, 0),
                       d.last_scanned_t
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
                path=row[1],
                config_path=row[2],
                image_count=row[3],
                has_caption=row[4],
                has_toml=row[5],
                last_scanned_t=row[6],
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
            dataset_row = conn.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,)).fetchone()
            if dataset_row is None:
                return ImagePage(images=[])

            dataset_id: int = dataset_row[0]

            rows = conn.execute(
                """
                SELECT id, file_name, path, has_caption, has_toml, draft_names, last_modified_t
                FROM dataset_images
                WHERE dataset_id = ? AND id > ?
                ORDER BY id
                LIMIT ?
                """,
                (dataset_id, after_id, limit + 1),
            ).fetchall()

            images = [
                ImageInfo(
                    id=row[0],
                    file_name=row[1],
                    path=row[2],
                    has_caption=bool(row[3]),
                    has_toml=bool(row[4]),
                    draft_names=row[5].split(",") if row[5] else [],
                    last_modified_t=row[6],
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
                SELECT di.id, di.file_name, di.path, di.has_caption, di.has_toml, di.draft_names, di.last_modified_t
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
                draft_names=row[5].split(",") if row[5] else [],
                last_modified_t=row[6],
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

        extras: dict[str, Any] = {}
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras = toml.load(f)
            except Exception:
                pass

        drafts: dict[str, str] = {}
        try:
            drafts = dataset_image.read_all_drafts()
        except Exception:
            pass

        return {"caption": caption, "extras": extras, "drafts": drafts}

    def update_caption(self, dataset_name: str, image_id: int, caption: str) -> bool:
        """Update the caption file for an image. Returns True on success."""
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        dataset_image = DatasetImage(path=str(image_path))
        dataset_image.update_caption(caption)

        # Update the index
        self._update_image_index(image_id, has_caption=True)
        return True

    # --- Dataset scanning / indexing ---

    def scan_dataset(self, name: str) -> bool:
        """Force a rescan of a dataset by name. Returns True if dataset was found."""
        conn = self._db.connection()
        try:
            row = conn.execute("SELECT id, path, config_path FROM datasets WHERE name = ?", (name,)).fetchone()
            if row is None:
                return False

            dataset_id, path, config_path = row[0], row[1], row[2]
            self._scan_dataset(conn, dataset_id, path, config_path)
            conn.commit()
            return True
        finally:
            conn.close()

    def register_dataset(self, path: str, *, name: str | None = None) -> DatasetInfo:
        """Register a dataset directory for indexing.

        Scans for a config.toml in the directory, indexes all images, and stores
        metadata in the DB.
        """
        dir_path = Path(path).resolve()
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        config_path = dir_path / "config.toml"
        config_path_str = str(config_path) if config_path.exists() else None

        # Derive name from directory if not provided
        if name is None:
            name = dir_path.name

        # Load config to get dataset entries
        config = self._load_config(config_path) if config_path.exists() else None

        conn = self._db.connection()
        try:
            # Upsert dataset record
            conn.execute(
                """
                INSERT INTO datasets (name, path, config_path, last_scanned_t, updated_t)
                VALUES (?, ?, ?, unixepoch(), unixepoch())
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name,
                    config_path = excluded.config_path,
                    updated_t = unixepoch()
                """,
                (name, str(dir_path), config_path_str),
            )

            row = conn.execute("SELECT id FROM datasets WHERE path = ?", (str(dir_path),)).fetchone()
            assert row is not None
            dataset_id = row[0]

            self._scan_dataset(conn, dataset_id, str(dir_path), config_path_str, config=config)
            conn.commit()
        finally:
            conn.close()

        result = self.get_dataset(name)
        assert result is not None
        return result

    def unregister_dataset(self, name: str) -> bool:
        """Remove a dataset and its images from the index. Returns True if found."""
        conn = self._db.connection()
        try:
            cursor = conn.execute("DELETE FROM datasets WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def discover_datasets(self, search_paths: list[str] | None = None) -> list[DatasetInfo]:
        """Scan search paths (and defaults) for dataset directories and register them.

        Looks for directories containing a ``config.toml`` file.
        """
        paths = list(search_paths or [])
        # Add some default search paths
        from yadc.cmd.app import CONFIG_PATH

        paths.append(str(CONFIG_PATH))

        found: list[DatasetInfo] = []
        seen_paths: set[str] = set()

        for search_path in paths:
            base = Path(search_path)
            if not base.is_dir():
                continue

            # Check direct children for config.toml
            for child in sorted(base.iterdir()):
                if not child.is_dir():
                    continue

                resolved = str(child.resolve())
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)

                config_file = child / "config.toml"
                if not config_file.exists():
                    continue

                try:
                    info = self.register_dataset(str(child))
                    found.append(info)
                except Exception as e:
                    self._logger.warning("Failed to register dataset at %s: %s", child, e)

        return found

    # --- Private helpers ---

    def _load_config(self, config_path: Path) -> Config | None:
        """Load and parse a dataset config file."""
        try:
            with open(config_path) as f:
                raw = toml.load(f)
            return parse_config(raw)
        except Exception as e:
            self._logger.warning("Failed to parse config at %s: %s", config_path, e)
            return None

    def _scan_dataset(
        self,
        conn: sqlite3.Connection,
        dataset_id: int,
        path: str,
        config_path: str | None,
        config: Config | None = None,
    ) -> None:
        """Scan a dataset directory and update the image index."""
        dir_path = Path(path)
        if not dir_path.is_dir():
            return

        # Load config if not provided
        if config is None and config_path:
            config = self._load_config(Path(config_path))

        # Determine image directories from config
        image_dirs: list[Path] = []
        if config:
            for entry in config.dataset:
                if entry.path:
                    p = Path(entry.path)
                    if not p.is_absolute():
                        p = dir_path / p
                    if p.is_dir():
                        image_dirs.append(p.resolve())

        # If no image dirs from config, use the dataset path itself
        if not image_dirs:
            image_dirs = [dir_path]

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

                disk_images[str(child)] = {
                    "file_name": child.name,
                    "has_caption": caption_path.exists(),
                    "has_toml": toml_path.exists(),
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
                    SET file_name = ?, has_caption = ?, has_toml = ?, draft_names = ?, last_modified_t = ?
                    WHERE id = ?
                    """,
                    (
                        meta["file_name"],
                        int(meta["has_caption"]),
                        int(meta["has_toml"]),
                        meta["draft_names"],
                        meta["last_modified_t"],
                        existing_by_path[img_path],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO dataset_images (dataset_id, path, file_name, has_caption, has_toml, draft_names, last_modified_t)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        img_path,
                        meta["file_name"],
                        int(meta["has_caption"]),
                        int(meta["has_toml"]),
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

    def _update_image_index(self, image_id: int, *, has_caption: bool | None = None) -> None:
        """Update specific fields on an indexed image."""
        conn = self._db.connection()
        try:
            sets: list[str] = []
            params: list[Any] = []
            if has_caption is not None:
                sets.append("has_caption = ?")
                params.append(int(has_caption))
            if not sets:
                return
            params.append(image_id)
            conn.execute(f"UPDATE dataset_images SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()
        finally:
            conn.close()

    def _refresh_stale_datasets(self, max_age_seconds: float = 60.0) -> None:
        """Rescan datasets that haven't been scanned recently."""
        conn = self._db.connection()
        try:
            cutoff = time.time() - max_age_seconds
            stale = conn.execute(
                "SELECT id, path, config_path FROM datasets WHERE last_scanned_t IS NULL OR last_scanned_t < ?",
                (cutoff,),
            ).fetchall()

            for dataset_id, path, config_path in stale:
                try:
                    self._scan_dataset(conn, dataset_id, path, config_path)
                    conn.commit()
                except Exception as e:
                    self._logger.warning("Failed to refresh dataset id=%d: %s", dataset_id, e)
        finally:
            conn.close()
