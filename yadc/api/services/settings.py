"""Settings service — simple key-value store backed by the SQLite settings table."""

import json
import sqlite3
from logging import Logger
from typing import Any

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service


class SettingsService(Service):
    """Read/write user settings from the SQLite settings table.

    Values are stored as JSON strings to preserve types (bools, ints, lists, etc.).
    """

    def __init__(self, db: DBConnectionFactory, logging: LoggingFactory):
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)

    def get(self, key: str, default: Any = None) -> Any:
        conn = self._db.connection()
        try:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            if row is None:
                return default
            return json.loads(row[0])
        except (json.JSONDecodeError, sqlite3.Error) as e:
            self._logger.warning("Failed to read setting %s: %s", key, e)
            return default
        finally:
            conn.close()

    def set(self, key: str, value: Any) -> None:
        conn = self._db.connection()
        try:
            conn.execute(
                "INSERT INTO settings (key, value, updated_t) VALUES (?, ?, unixepoch()) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_t = excluded.updated_t",
                (key, json.dumps(value)),
            )
            conn.commit()
        except sqlite3.Error as e:
            self._logger.error("Failed to write setting %s: %s", key, e)
        finally:
            conn.close()

    def delete(self, key: str) -> bool:
        conn = self._db.connection()
        try:
            cursor = conn.execute("DELETE FROM settings WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            self._logger.error("Failed to delete setting %s: %s", key, e)
            return False
        finally:
            conn.close()

    def list_all(self) -> dict[str, Any]:
        conn = self._db.connection()
        try:
            rows = conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
            result: dict[str, Any] = {}
            for key, value_json in rows:
                try:
                    result[key] = json.loads(value_json)
                except json.JSONDecodeError:
                    result[key] = value_json
            return result
        except sqlite3.Error as e:
            self._logger.error("Failed to list settings: %s", e)
            return {}
        finally:
            conn.close()
