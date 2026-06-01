from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from ..configuration import Configuration
from .db_migrations import DBMigrations
from .logging_factory import LoggingFactory
from .service import Service


class _Transaction:
    """Manages a single connection with nested savepoint support."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn: sqlite3.Connection = conn
        self._depth: int = 0

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def begin(self) -> None:
        if self._depth == 0:
            self._conn.execute("BEGIN")
        else:
            self._conn.execute(f"SAVEPOINT sp_{self._depth}")
        self._depth += 1

    def commit(self) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._conn.commit()
        else:
            self._conn.execute(f"RELEASE SAVEPOINT sp_{self._depth}")

    def rollback(self) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._conn.rollback()
        else:
            self._conn.execute(f"ROLLBACK TO SAVEPOINT sp_{self._depth}")

    def close(self) -> None:
        self._conn.close()


class DBConnectionFactory(Service):
    def __init__(self, configuration: Configuration, logging: LoggingFactory, migrations: DBMigrations) -> None:
        self.path: str = configuration.db_path
        self._logger: logging.Logger = logging.get_logger(__name__)
        self._active_transaction: ContextVar[_Transaction | None] = ContextVar("_active_transaction", default=None)

        self._init_db(migrations)

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
        """Returns a connection.

        If there is an active transaction in the current context (thread or
        asyncio task), returns the transaction's connection.  Otherwise creates
        a new standalone connection (legacy behaviour).
        """
        txn = self._active_transaction.get()
        if txn is not None:
            return txn.connection
        return self._connection()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a connection within a transaction.

        Nesting is supported: inner calls create SQLite savepoints.
        The connection is also returned by ``connection()`` while the
        transaction is active, so existing callers are automatically
        enrolled without changes.
        """
        existing = self._active_transaction.get()
        if existing is not None:
            # Nested: open a savepoint within the existing transaction.
            existing.begin()
            try:
                yield existing.connection
            except BaseException:
                existing.rollback()
                raise
            else:
                existing.commit()
            return

        # Outermost: open a fresh connection and BEGIN.
        conn = self._connection()
        txn = _Transaction(conn)
        token = self._active_transaction.set(txn)
        txn.begin()
        try:
            yield conn
        except BaseException:
            txn.rollback()
            raise
        else:
            txn.commit()
        finally:
            self._active_transaction.reset(token)
            txn.close()
