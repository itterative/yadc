"""Config revision history — track TOML snapshots in SQLite for undo/restore."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from logging import Logger

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service

# Default max history entries per dataset before pruning kicks in.
DEFAULT_MAX_ENTRIES = 50


@dataclass
class ConfigHistoryEntry:
    """A single config revision snapshot."""

    id: int
    dataset_name: str
    content: str
    created_t: float


class ConfigHistoryService(Service):
    """Stores and retrieves config TOML snapshots for revision history.

    On every config write (PATCH/PUT), callers should call :meth:`save_snapshot`
    *before* writing the new content. This captures the pre-write state so it
    can be restored later.
    """

    def __init__(self, db: DBConnectionFactory, logging: LoggingFactory) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)

    def save_snapshot(self, dataset_name: str, content: str) -> int:
        """Save a config snapshot and return its row ID.

        Prunes old entries if the dataset exceeds ``max_entries``.
        """
        conn = self._db.connection()
        try:
            cursor = conn.execute(
                "INSERT INTO config_history (dataset_name, content, created_t) VALUES (?, ?, ?)",
                (dataset_name, content, time.time()),
            )
            row_id = cursor.lastrowid
            assert row_id is not None
            conn.commit()

            self._prune(conn, dataset_name)
            self._logger.debug("Saved config snapshot for '%s' (id=%d).", dataset_name, row_id)
            return row_id
        finally:
            conn.close()

    def list_history(self, dataset_name: str, *, limit: int = 50, before_id: int | None = None) -> list[ConfigHistoryEntry]:
        """List history entries for a dataset, most recent first.

        Supports cursor-based pagination via ``before_id``.
        """
        conn = self._db.connection()
        try:
            if before_id is not None:
                rows = conn.execute(
                    "SELECT id, dataset_name, content, created_t FROM config_history WHERE dataset_name = ? AND id < ? ORDER BY id DESC LIMIT ?",
                    (dataset_name, before_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, dataset_name, content, created_t FROM config_history WHERE dataset_name = ? ORDER BY id DESC LIMIT ?",
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
        finally:
            conn.close()

    def get_entry(self, entry_id: int) -> ConfigHistoryEntry | None:
        """Get a single history entry by ID."""
        conn = self._db.connection()
        try:
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
        finally:
            conn.close()

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a single history entry. Returns True if found and deleted."""
        conn = self._db.connection()
        try:
            cursor = conn.execute("DELETE FROM config_history WHERE id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def _prune(self, conn: sqlite3.Connection, dataset_name: str) -> None:
        """Remove oldest entries if count exceeds DEFAULT_MAX_ENTRIES."""
        count = conn.execute(
            "SELECT COUNT(*) FROM config_history WHERE dataset_name = ?",
            (dataset_name,),
        ).fetchone()[0]

        if count <= DEFAULT_MAX_ENTRIES:
            return

        excess = count - DEFAULT_MAX_ENTRIES
        conn.execute(
            "DELETE FROM config_history WHERE dataset_name = ? AND id IN (SELECT id FROM config_history WHERE dataset_name = ? ORDER BY id ASC LIMIT ?)",
            (dataset_name, dataset_name, excess),
        )
        conn.commit()
        self._logger.debug("Pruned %d old config history entries for '%s'.", excess, dataset_name)
