from __future__ import annotations

import logging
import sqlite3
import uuid
from threading import Thread

from ..configuration import Configuration
from .db_migrations import DBMigrations
from .logging_factory import LoggingFactory
from .service import Service


class DBConnectionFactory(Service):
    def __init__(self, configuration: Configuration, logging: LoggingFactory, migrations: DBMigrations) -> None:
        self.path: str = configuration.db_path
        self._logger: logging.Logger = logging.get_logger(__name__)

        self._init_thread: Thread = Thread(target=self._init_db, args=(migrations,), daemon=True)
        self._init_thread.start()

    def _init_db(self, migrations: DBMigrations):
        self._logger.debug("Initializing database at %s", self.path)

        with self._connection() as conn:
            migrations.run_migrations(conn)

        self._logger.debug("Finished database initialization.")

    @staticmethod
    def _sql_uuid():
        return uuid.uuid4().hex

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma foreign_keys=on")
        conn.execute("pragma busy_timeout=5000")

        conn.create_function("uuid", 0, self._sql_uuid, deterministic=False)

        return conn

    def connection(self) -> sqlite3.Connection:
        """Returns a new connection, waiting for init to complete first."""
        self._init_thread.join()
        return self._connection()
