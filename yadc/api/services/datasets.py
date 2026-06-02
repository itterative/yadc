"""Dataset service — filesystem scanning, SQLite indexing, and image queries.

A "dataset" is a named yadc config TOML stored in the XDG state directory
(``STATE_PATH/<name>/config.toml``). The TOML is the source of truth — it
defines API settings, prompts, and ``[[dataset]]`` entries pointing to image
directories.

Two registration flows:

- **Import**: copy an existing TOML into the state dir (resolving relative
  paths to absolute first).
- **Create**: write a fresh TOML with the given image paths.

Both end up as ``STATE_PATH/<name>/config.toml``.
"""

import sqlite3
import time
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any

import toml
from PIL import Image

from yadc.cmd.app import STATE_PATH
from yadc.core.config import Config, parse_config
from yadc.core.dataset import DatasetImage

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service

# Image extensions we recognize (matching what PIL can open).
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".ico"})


def _dataset_state_dir(name: str) -> Path:
    """Return the state directory for a named dataset."""
    return STATE_PATH / name


def _dataset_config_path(name: str) -> Path:
    """Return the config TOML path for a named dataset."""
    return _dataset_state_dir(name) / "config.toml"


@dataclass
class DatasetInfo:
    """Summary of a dataset as returned by the listing API."""

    name: str
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
    width: int = 0
    height: int = 0
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
        # Ensure state dir exists
        STATE_PATH.mkdir(parents=True, exist_ok=True)

    # --- Dataset listing ---

    def list_datasets(self) -> list[DatasetInfo]:
        """Return all indexed datasets, refreshing stale entries first."""
        self._refresh_stale_datasets()

        conn = self._db.connection()
        try:
            rows = conn.execute(
                """
                SELECT d.name, d.config_path,
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
                    config_path=row[1],
                    image_count=row[2],
                    has_caption=row[3],
                    has_toml=row[4],
                    last_scanned_t=row[5],
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
                SELECT d.name, d.config_path,
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
                config_path=row[1],
                image_count=row[2],
                has_caption=row[3],
                has_toml=row[4],
                last_scanned_t=row[5],
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
                SELECT id, file_name, path, has_caption, has_toml, width, height, draft_names, last_modified_t
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
                    width=row[5] or 0,
                    height=row[6] or 0,
                    draft_names=row[7].split(",") if row[7] else [],
                    last_modified_t=row[8],
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
                SELECT di.id, di.file_name, di.path, di.has_caption, di.has_toml, di.width, di.height, di.draft_names, di.last_modified_t
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

        extras_raw: str = ""
        extras: dict[str, Any] = {}
        if dataset_image.toml_path.exists():
            try:
                with open(dataset_image.toml_path) as f:
                    extras_raw = f.read()
                    f.seek(0)
                    extras = toml.loads(extras_raw)
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
                    extras = toml.loads(f.read())
            except Exception:
                pass

        # Apply extras as additional fields on the DatasetImage
        if extras:
            dataset_image = DatasetImage.model_validate({"path": str(image_path), **extras})

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
        }

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

    def update_extras(self, dataset_name: str, image_id: int, extras_raw: str) -> bool:
        """Update the TOML extras sidecar for an image. Returns True on success."""
        info = self.get_image(dataset_name, image_id)
        if info is None:
            return False

        image_path = Path(info.path)
        if not image_path.exists():
            return False

        # Validate TOML before writing
        try:
            toml.loads(extras_raw)
        except Exception as e:
            raise ValueError(f"Invalid TOML: {e}") from e

        dataset_image = DatasetImage(path=str(image_path))
        dataset_image.toml_path.write_text(extras_raw)

        self._update_image_index(image_id, has_toml=bool(extras_raw.strip()))
        return True

    # --- Dataset registration ---

    def import_dataset(self, name: str, toml_path: str) -> DatasetInfo:
        """Import an existing TOML config as a named dataset.

        Copies the TOML to ``STATE_PATH/<name>/config.toml``, resolving any
        relative paths in ``[[dataset]]`` entries to absolute first.
        """
        source = Path(toml_path).resolve()
        if not source.is_file():
            raise ValueError(f"Not a file: {source}")

        # Load and resolve relative paths
        raw = self._load_raw_config(source)
        raw = self._resolve_relative_paths(raw, source.parent)

        # Write to state dir
        dest_dir = _dataset_state_dir(name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _dataset_config_path(name)
        with open(dest, "w") as f:
            toml.dump(raw, f)

        return self._register(name, str(dest))

    def create_dataset(self, name: str, image_paths: list[str]) -> DatasetInfo:
        """Create a new dataset with the given image directories.

        Writes a minimal TOML to ``STATE_PATH/<name>/config.toml`` with one
        ``[[dataset]]`` entry per path.
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

        dest_dir = _dataset_state_dir(name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _dataset_config_path(name)
        with open(dest, "w") as f:
            toml.dump(raw, f)

        return self._register(name, str(dest))

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
            return True
        finally:
            conn.close()

    def unregister_dataset(self, name: str) -> bool:
        """Remove a dataset, its images from the index, and its state dir."""
        import shutil

        conn = self._db.connection()
        try:
            cursor = conn.execute("DELETE FROM datasets WHERE name = ?", (name,))
            conn.commit()
            found = cursor.rowcount > 0
        finally:
            conn.close()

        if found:
            state_dir = _dataset_state_dir(name)
            if state_dir.is_dir():
                shutil.rmtree(state_dir, ignore_errors=True)

        return found

    def _register(self, name: str, config_path: str) -> DatasetInfo:
        """Upsert a dataset record and scan its images."""
        config = self._load_config(Path(config_path))

        conn = self._db.connection()
        try:
            conn.execute(
                """
                INSERT INTO datasets (name, config_path, last_scanned_t, updated_t)
                VALUES (?, ?, unixepoch(), unixepoch())
                ON CONFLICT(name) DO UPDATE SET
                    config_path = excluded.config_path,
                    updated_t = unixepoch()
                """,
                (name, config_path),
            )

            row = conn.execute("SELECT id FROM datasets WHERE name = ?", (name,)).fetchone()
            assert row is not None
            dataset_id = row[0]

            self._scan_dataset(conn, dataset_id, config_path, config=config)
            conn.commit()
        finally:
            conn.close()

        result = self.get_dataset(name)
        assert result is not None
        return result

    # --- Private helpers ---

    def _load_raw_config(self, config_path: Path) -> dict[str, Any]:
        """Load a TOML config as a raw dict."""
        with open(config_path) as f:
            return toml.load(f)

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
                raw = toml.load(f)
            return parse_config(raw, strict=False)
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

    def _refresh_stale_datasets(self, max_age_seconds: float = 60.0) -> None:
        """Rescan datasets that haven't been scanned recently."""
        conn = self._db.connection()
        try:
            cutoff = time.time() - max_age_seconds
            stale = conn.execute(
                "SELECT id, config_path FROM datasets WHERE last_scanned_t IS NULL OR last_scanned_t < ?",
                (cutoff,),
            ).fetchall()

            for dataset_id, config_path in stale:
                try:
                    self._scan_dataset(conn, dataset_id, config_path)
                    conn.commit()
                except Exception as e:
                    self._logger.warning("Failed to refresh dataset id=%d: %s", dataset_id, e)
        finally:
            conn.close()
