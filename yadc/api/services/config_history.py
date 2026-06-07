"""Config revision history — track TOML snapshots in SQLite for undo/restore.

The SQL lives in :class:`ConfigHistoryRepository`. This service owns:

- Public Python API (re-exports :class:`ConfigHistoryEntry` from the repo)
- Transaction boundaries for multi-statement work
- Business logic (dataset validation, logging, prune policy)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from logging import Logger

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .config_history_repository import ConfigHistoryEntry, ConfigHistoryRepository
from .dataset_repository import DatasetRepository

# Re-export the data model so the public service API is unchanged.
__all__ = ["ConfigHistoryEntry", "ConfigHistoryPage", "ConfigHistoryService"]


@dataclass
class ConfigHistoryPage:
    """A paginated page of config history entries.

    ``next_token`` is the opaque cursor to pass back for the next
    (older) page. ``None`` means this is the last page. The
    encoding is an implementation detail — the client treats it as
    a string token.
    """

    entries: list[ConfigHistoryEntry]
    next_token: str | None = None


# Default max history entries per dataset before pruning kicks in.
DEFAULT_MAX_ENTRIES = 50


class ConfigHistoryService(Service):
    """Stores and retrieves config TOML snapshots for revision history.

    On every config write (PATCH/PUT), callers should call :meth:`save_snapshot`
    *before* writing the new content. This captures the pre-write state so it
    can be restored later.
    """

    def __init__(
        self,
        db: DBConnectionFactory,
        logging: LoggingFactory,
        repo: ConfigHistoryRepository,
        datasets: DatasetRepository,
    ) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)
        self._repo: ConfigHistoryRepository = repo
        self._datasets: DatasetRepository = datasets

    def save_snapshot(self, dataset_name: str, content: str) -> int:
        """Save a config snapshot and return its row ID.

        Prunes old entries if the dataset exceeds ``DEFAULT_MAX_ENTRIES``.
        """
        # Look up dataset_id — skip snapshot if dataset isn't registered yet.
        row = self._datasets.get_dataset_row(dataset_name)
        if row is None:
            self._logger.warning("Skipping config snapshot for '%s': dataset not registered in DB.", dataset_name)
            return -1
        dataset_id = row[0]

        with self._db.transaction():
            row_id = self._repo.insert_snapshot(dataset_name, dataset_id, content, time.time())
            self._repo.prune_old(dataset_name, DEFAULT_MAX_ENTRIES)

        self._logger.debug("Saved config snapshot for '%s' (id=%d).", dataset_name, row_id)
        return row_id

    def list_history(self, dataset_name: str, *, limit: int = 50, next: str | None = None) -> ConfigHistoryPage:
        """List history entries for a dataset, most recent first.

        Args:
            dataset_name: Name of the dataset.
            limit: Max entries to return.
            next: Opaque pagination token returned by a previous
                call as ``next_token``. ``None`` (the default) means
                "first page" — returns the newest entries. The
                client treats this as a string token; the server
                decodes it to determine the next page.

        Returns:
            A :class:`ConfigHistoryPage` with the entries (newest
            first) and a ``next_token`` for the next (older) page,
            or ``None`` if this is the last page.
        """
        before_id = self._decode_next_token(next)
        # Fetch limit+1 to detect "is there a next page" without an
        # extra count query.
        rows = self._repo.list_history(dataset_name, limit=limit + 1, before_id=before_id)
        has_more = len(rows) > limit
        entries = rows[:limit]
        next_token = str(entries[-1].id) if has_more and entries else None
        return ConfigHistoryPage(entries=entries, next_token=next_token)

    @staticmethod
    def _decode_next_token(token: str | None) -> int:
        """Decode an opaque ``next`` token into the repo's ``before_id`` cursor.

        ``None`` or an empty token means "no upper bound" — use a
        value larger than any real id so all rows pass the
        ``id < ?`` filter. SQLite's INTEGER is 64-bit; ``2**63 - 1``
        is the max signed value and won't be reached by
        AUTOINCREMENT for billions of years.
        """
        if not token:
            return 2**63 - 1
        return int(token)

    def get_entry(self, entry_id: int) -> ConfigHistoryEntry | None:
        """Get a single history entry by ID."""
        return self._repo.get_entry(entry_id)

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a single history entry. Returns True if found and deleted."""
        return self._repo.delete_entry(entry_id)
