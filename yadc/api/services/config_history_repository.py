"""SQL access for the ``config_history`` table.

The repository owns:

- The data model: :class:`ConfigHistoryEntry` is stored here because
  it's shaped by the SQL.
- All SQL: query text, parameter binding, row-to-dataclass mapping.

Connection management is internal to each public method — callers do
not pass a connection. Every method uses
:meth:`DBConnectionFactory.connection`, which auto-enrolls in any
active ``with db.transaction():`` block, so multi-statement
orchestration can be done at the service level by wrapping several
repo calls in a single transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service


@dataclass
class ConfigHistoryEntry:
    """A single config revision snapshot."""

    id: int
    dataset_name: str
    content: str
    created_t: float


class ConfigHistoryRepository(Service):
    """Read/write operations on the SQLite ``config_history`` table."""

    def __init__(self, db: DBConnectionFactory, logging: LoggingFactory) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)

    # --- Reads ---

    def list_history(self, dataset_name: str, *, limit: int = 50, before_id: int | None = None) -> list[ConfigHistoryEntry]:
        """List history entries for a dataset, most recent first.

        Supports cursor-based pagination via ``before_id``.
        """
        with self._db.connection() as conn:
            if before_id is not None:
                rows = conn.execute(
                    "SELECT id, dataset_name, content, created_t "
                    "FROM config_history WHERE dataset_name = ? AND id < ? "
                    "ORDER BY id DESC LIMIT ?",
                    (dataset_name, before_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, dataset_name, content, created_t "
                    "FROM config_history WHERE dataset_name = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (dataset_name, limit),
                ).fetchall()

        return [
            ConfigHistoryEntry(
                id=row[0],
                dataset_name=row[1],
                content=row[2],
                created_t=row[3],
            )
            for row in rows
        ]

    def get_entry(self, entry_id: int) -> ConfigHistoryEntry | None:
        """Get a single history entry by ID."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT id, dataset_name, content, created_t FROM config_history WHERE id = ?",
                (entry_id,),
            ).fetchone()
        if row is None:
            return None
        return ConfigHistoryEntry(
            id=row[0],
            dataset_name=row[1],
            content=row[2],
            created_t=row[3],
        )

    def count_entries(self, dataset_name: str) -> int:
        """Return the number of history entries for a dataset."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM config_history WHERE dataset_name = ?",
                (dataset_name,),
            ).fetchone()
        return row[0]

    # --- Writes ---

    def insert_snapshot(self, dataset_name: str, dataset_id: int, content: str, created_t: float) -> int:
        """Insert a config snapshot row and return its ID."""
        with self._db.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO config_history (dataset_name, dataset_id, content, created_t) VALUES (?, ?, ?, ?)",
                (dataset_name, dataset_id, content, created_t),
            )
            row_id = cursor.lastrowid
            assert row_id is not None
            return row_id

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a single history entry. Returns True if found and deleted."""
        with self._db.connection() as conn:
            cursor = conn.execute("DELETE FROM config_history WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0

    def prune_old(self, dataset_name: str, keep_count: int) -> int:
        """Remove oldest entries beyond ``keep_count``. Returns the number of deleted rows."""
        with self._db.connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM config_history WHERE dataset_name = ?",
                (dataset_name,),
            ).fetchone()[0]

            if count <= keep_count:
                return 0

            excess = count - keep_count
            conn.execute(
                "DELETE FROM config_history WHERE dataset_name = ? "
                "AND id IN (SELECT id FROM config_history WHERE dataset_name = ? ORDER BY id ASC LIMIT ?)",
                (dataset_name, dataset_name, excess),
            )
            self._logger.debug("Pruned %d old config history entries for '%s'.", excess, dataset_name)
            return excess
