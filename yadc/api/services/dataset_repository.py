"""SQL access for the ``datasets`` and ``dataset_images`` tables.

The repository owns:

- The data model: :class:`DatasetInfo` and :class:`ImageInfo` are stored
  here because they're shaped by the SQL.
- All SQL: query text, parameter binding, row-to-dataclass mapping.

Connection management is internal to each public method — callers do
not pass a connection. Every method uses
:meth:`DBConnectionFactory.connection`, which auto-enrolls in any
active ``with db.transaction():`` block, so multi-statement
orchestration can be done at the service level by wrapping several
repo calls in a single transaction.

The service layer owns:

- Business logic (rescan policy, watcher integration, file I/O).
- The ``with db.transaction():`` boundary for multi-statement work.
- Multi-method orchestration that composes repo calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import Logger
from typing import Any

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .managed_paths import compute_delete_path


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


class DatasetRepository(Service):
    """Read/write operations on the datasets and dataset_images tables."""

    def __init__(self, db: DBConnectionFactory, logging: LoggingFactory) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)

    # --- Datasets: reads ---

    def list_datasets(self) -> list[DatasetInfo]:
        """Return all datasets with their aggregated image stats."""
        with self._db.connection() as conn:
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

    def get_dataset(self, name: str) -> DatasetInfo | None:
        """Return a single dataset by name, or None if not registered."""
        with self._db.connection() as conn:
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

    def get_dataset_row(self, name: str) -> tuple[int, str | None] | None:
        """Return (id, config_path) for a dataset, or None if not registered."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT id, config_path FROM datasets WHERE name = ?", (name,)).fetchone()
        return (row[0], row[1]) if row is not None else None

    def list_stale(self, cutoff_t: float) -> list[tuple[int, str, str | None]]:
        """Return (id, name, config_path) for datasets that have never been scanned or were scanned before ``cutoff_t``."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT id, name, config_path FROM datasets WHERE last_scanned_t IS NULL OR last_scanned_t < ?",
                (cutoff_t,),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def list_all_for_watcher(self) -> list[tuple[str, str | None]]:
        """Return (name, config_path) for every registered dataset."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT name, config_path FROM datasets").fetchall()
        return [(r[0], r[1]) for r in rows]

    # --- Dataset images: reads ---

    def list_images(
        self,
        dataset_name: str,
        *,
        after_id: int,
        limit: int,
    ) -> list[ImageInfo]:
        """Return up to ``limit`` images for a dataset, paginated by id."""
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT di.id, di.file_name, di.path, di.has_caption, di.has_toml,
                       di.width, di.height, di.draft_names, di.last_modified_t,
                       d.config_path
                FROM dataset_images di
                JOIN datasets d ON d.id = di.dataset_id
                WHERE d.name = ? AND di.id > ?
                ORDER BY di.id
                LIMIT ?
                """,
                (dataset_name, after_id, limit),
            ).fetchall()
        return [
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
                delete_path=compute_delete_path(row[2], row[9]),
            )
            for row in rows
        ]

    def get_image(self, dataset_name: str, image_id: int) -> ImageInfo | None:
        """Return a single image by id within a dataset, or None."""
        with self._db.connection() as conn:
            row = conn.execute(
                """
                SELECT di.id, di.file_name, di.path, di.has_caption, di.has_toml,
                       di.width, di.height, di.draft_names, di.last_modified_t,
                       d.config_path
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
            delete_path=compute_delete_path(row[2], row[9]),
        )

    def get_image_by_path(self, dataset_name: str, image_path: str) -> ImageInfo | None:
        """Return an image by its on-disk path, or None."""
        with self._db.connection() as conn:
            row = conn.execute(
                """
                SELECT di.id, di.file_name, di.path, di.has_caption, di.has_toml,
                       di.width, di.height, di.draft_names, di.last_modified_t,
                       d.config_path
                FROM dataset_images di
                JOIN datasets d ON d.id = di.dataset_id
                WHERE d.name = ? AND di.path = ?
                """,
                (dataset_name, image_path),
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
            delete_path=compute_delete_path(row[2], row[9]),
        )

    def get_image_path(self, dataset_name: str, image_id: int) -> str | None:
        """Return the on-disk path for an image, or None if not indexed."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT di.path FROM dataset_images di JOIN datasets d ON d.id = di.dataset_id WHERE d.name = ? AND di.id = ?",
                (dataset_name, image_id),
            ).fetchone()
        return row[0] if row is not None else None

    def list_draft_names_csv(self, dataset_name: str) -> list[str]:
        """Return distinct non-empty ``draft_names`` CSV strings for a dataset.

        The service splits each CSV into individual names and unions the
        results — splitting is policy, not storage shape.
        """
        with self._db.connection() as conn:
            row = conn.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,)).fetchone()
            if row is None:
                return []
            dataset_id: int = row[0]
            rows = conn.execute(
                """
                SELECT DISTINCT draft_names FROM dataset_images
                WHERE dataset_id = ? AND draft_names != ''
                """,
                (dataset_id,),
            ).fetchall()
        return [csv for (csv,) in rows]

    def list_image_paths(self, dataset_id: int) -> dict[str, int]:
        """Return (path → id) for every image in a dataset.

        Used by the service's scan diff orchestration: it reads the
        current set of paths, computes the diff against the disk state,
        then issues ``upsert_image`` / ``delete_image`` calls inside the
        same transaction.
        """
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT id, path FROM dataset_images WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchall()
        return {row[1]: row[0] for row in rows}

    # --- Datasets: writes ---

    def upsert_dataset(self, name: str, config_path: str, source: str) -> int:
        """Insert or update a dataset row. Returns the dataset id."""
        with self._db.connection() as conn:
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
            return row[0]

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset row (cascades to its images). Returns True if a row was deleted."""
        with self._db.connection() as conn:
            cursor = conn.execute("DELETE FROM datasets WHERE name = ?", (name,))
            return cursor.rowcount > 0

    def update_dataset_stats(self, dataset_id: int, image_count: int) -> None:
        """Refresh the dataset's image_count and last_scanned_t."""
        with self._db.connection() as conn:
            conn.execute(
                """
                UPDATE datasets
                SET image_count = ?, last_scanned_t = unixepoch(), updated_t = unixepoch()
                WHERE id = ?
                """,
                (image_count, dataset_id),
            )

    # --- Dataset images: writes ---

    def upsert_image(
        self,
        *,
        dataset_id: int,
        path: str,
        file_name: str,
        has_caption: bool,
        has_toml: bool,
        width: int,
        height: int,
        draft_names: str,
        last_modified_t: float | None,
    ) -> None:
        """Insert a new image, or update the existing one with the same path."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO dataset_images (dataset_id, path, file_name, has_caption, has_toml,
                                            width, height, draft_names, last_modified_t)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_name = excluded.file_name,
                    has_caption = excluded.has_caption,
                    has_toml = excluded.has_toml,
                    width = excluded.width,
                    height = excluded.height,
                    draft_names = excluded.draft_names,
                    last_modified_t = excluded.last_modified_t
                """,
                (
                    dataset_id,
                    path,
                    file_name,
                    int(has_caption),
                    int(has_toml),
                    width,
                    height,
                    draft_names,
                    last_modified_t,
                ),
            )

    def delete_image(self, image_id: int) -> None:
        """Delete an image row by id."""
        with self._db.connection() as conn:
            conn.execute("DELETE FROM dataset_images WHERE id = ?", (image_id,))

    def update_image_flags(
        self,
        image_id: int,
        *,
        has_caption: bool | None = None,
        has_toml: bool | None = None,
    ) -> None:
        """Update specific boolean flags on an image row. No-op if nothing to set."""
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
        with self._db.connection() as conn:
            conn.execute(f"UPDATE dataset_images SET {', '.join(sets)} WHERE id = ?", params)

    def refresh_image_disk_state(
        self,
        image_id: int,
        *,
        has_caption: bool,
        has_toml: bool,
        draft_names: str,
    ) -> None:
        """Update an image's caption/toml/draft state from a fresh disk read."""
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE dataset_images SET has_caption = ?, has_toml = ?, draft_names = ? WHERE id = ?",
                (int(has_caption), int(has_toml), draft_names, image_id),
            )
