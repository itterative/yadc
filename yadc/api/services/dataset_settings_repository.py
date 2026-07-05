"""SQL access for the ``dataset_settings`` table.

The repository owns:

- All SQL for the ``dataset_settings`` table (query text, parameter binding).
- Connection / transaction boundaries: every public method calls
  :meth:`DBConnectionFactory.connection` (for reads) or
  :meth:`DBConnectionFactory.transaction` (for writes).

The service layer (:mod:`yadc.api.services.tag_policy_service`) owns:

- Encoding / decoding of JSON values.
- Public Python types (e.g. :class:`TagPolicy`).
- Error handling and logging.

Schema (see migration ``0009_dataset_settings``): one row per
``(dataset_id, key)`` pair. ``value`` is opaque from the repository's
point of view — callers JSON-encode / JSON-decode at the boundary. The
``ON DELETE CASCADE`` on the FK is the dataset cleanup story; no extra
service code is needed.
"""

from __future__ import annotations

from logging import Logger

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service


class DatasetSettingsRepository(Service):
    """Read/write operations on the ``dataset_settings`` table.

    Connection and transaction management is internal — callers do not
    pass a connection. Values are stored as raw JSON strings; encoding
    and decoding happens in the service layer.
    """

    def __init__(self, db: DBConnectionFactory, logging: LoggingFactory) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)

    def get(self, dataset_id: int, key: str) -> str | None:
        """Return the raw JSON-encoded value for ``(dataset_id, key)``, or ``None`` if absent."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT value FROM dataset_settings WHERE dataset_id = ? AND key = ?",
                (dataset_id, key),
            ).fetchone()
        return row[0] if row is not None else None

    def upsert(self, dataset_id: int, key: str, value_json: str) -> None:
        """Insert or update ``(dataset_id, key)`` with the given JSON-encoded value."""
        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO dataset_settings (dataset_id, key, value, updated_t) VALUES (?, ?, ?, unixepoch()) "
                "ON CONFLICT(dataset_id, key) DO UPDATE SET value = excluded.value, updated_t = excluded.updated_t",
                (dataset_id, key, value_json),
            )

    def delete(self, dataset_id: int, key: str) -> int:
        """Delete ``(dataset_id, key)``. Returns the number of rows deleted (0 or 1)."""
        with self._db.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM dataset_settings WHERE dataset_id = ? AND key = ?",
                (dataset_id, key),
            )
            return cursor.rowcount

    def list_for_dataset(self, dataset_id: int) -> list[tuple[str, str]]:
        """Return all ``(key, value_json)`` pairs for ``dataset_id``, ordered by key."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM dataset_settings WHERE dataset_id = ? ORDER BY key",
                (dataset_id,),
            ).fetchall()
        return [(row[0], row[1]) for row in rows]
