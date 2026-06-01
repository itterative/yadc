"""Settings service — JSON-typed KV store backed by SQLite.

The SQL lives in :class:`SettingsRepository`. This service owns:

- Public Python types (decoded JSON values)
- Error handling and logging
"""

from __future__ import annotations

import json
import sqlite3
from logging import Logger
from typing import Any

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .settings_repository import SettingsRepository


class SettingsService(Service):
    """Read/write user settings from the SQLite settings table.

    Values are stored as JSON strings to preserve types (bools, ints, lists, etc.).
    """

    def __init__(
        self,
        db: DBConnectionFactory,
        logging: LoggingFactory,
        repo: SettingsRepository,
    ) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)
        self._repo: SettingsRepository = repo

    def get(self, key: str, default: Any = None) -> Any:
        try:
            value_json = self._repo.get(key)
        except sqlite3.Error as e:
            self._logger.warning("Failed to read setting %s: %s", key, e)
            return default

        if value_json is None:
            return default
        try:
            return json.loads(value_json)
        except json.JSONDecodeError:
            # Preserve the previous behaviour: corrupted JSON falls back to
            # the default rather than raising.
            return default

    def set(self, key: str, value: Any) -> None:
        try:
            self._repo.upsert(key, json.dumps(value))
        except sqlite3.Error as e:
            self._logger.error("Failed to write setting %s: %s", key, e)

    def delete(self, key: str) -> bool:
        try:
            return self._repo.delete(key) > 0
        except sqlite3.Error as e:
            self._logger.error("Failed to delete setting %s: %s", key, e)
            return False

    def list_all(self) -> dict[str, Any]:
        try:
            rows = self._repo.list_all()
        except sqlite3.Error as e:
            self._logger.error("Failed to list settings: %s", e)
            return {}

        result: dict[str, Any] = {}
        for key, value_json in rows:
            try:
                result[key] = json.loads(value_json)
            except json.JSONDecodeError:
                result[key] = value_json
        return result
