"""SQL access for the settings table.

The repository owns:

- All SQL for the ``settings`` table (query text, parameter binding).
- Connection / transaction boundaries: every public method calls
  :meth:`DBConnectionFactory.connection` (for reads) or
  :meth:`DBConnectionFactory.transaction` (for writes).

The service (:class:`SettingsService`) owns:

- Public Python types (decoded JSON values)
- Error handling and logging
"""

from __future__ import annotations

from logging import Logger

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service


class SettingsRepository(Service):
    """Read/write operations on the SQLite ``settings`` table.

    Connection and transaction management is internal — callers do not
    pass a connection. Values are stored as raw JSON strings; encoding
    and decoding happens in the service layer.
    """

    def __init__(self, db: DBConnectionFactory, logging: LoggingFactory) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)

    def get(self, key: str) -> str | None:
        """Return the raw JSON-encoded value for ``key``, or ``None`` if not present."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row[0] if row is not None else None

    def upsert(self, key: str, value_json: str) -> None:
        """Insert or update a key with the given JSON-encoded value."""
        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO settings (key, value, updated_t) VALUES (?, ?, unixepoch()) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_t = excluded.updated_t",
                (key, value_json),
            )

    def delete(self, key: str) -> int:
        """Delete a key. Returns the number of rows deleted (0 or 1)."""
        with self._db.connection() as conn:
            cursor = conn.execute("DELETE FROM settings WHERE key = ?", (key,))
            return cursor.rowcount

    def list_all(self) -> list[tuple[str, str]]:
        """Return all (key, value_json) pairs ordered by key."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
        return [(row[0], row[1]) for row in rows]
