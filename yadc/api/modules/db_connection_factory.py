"""``DBConnectionFactory`` — SQLite WAL/foreign-key connections with auto-enrolling context managers.

The factory is constructed once per app from ``Configuration.db_path``,
``LoggingFactory``, and ``DBMigrations``; ``_init_db`` opens a
throwaway connection and runs any pending migrations synchronously so
the first request doesn't pay the migration cost.

``_connection()`` opens a fresh ``sqlite3.Connection`` with
``pragma journal_mode=wal``, ``pragma foreign_keys=on``,
``pragma busy_timeout=5000``, and a custom ``uuid()`` SQL function
implemented in Python (returns a hex ``uuid4``) for use as a default
row id.

The public ``connection()`` context manager auto-enrols in any active
``transaction()`` in the current context (a ``ContextVar`` ensures
async tasks and threads don't share state by accident). If no
transaction is active, a new connection is opened, the body is run,
the connection is committed on clean exit and closed on exception.
The public ``transaction()`` wraps a connection in ``BEGIN``/``END``;
nesting is supported via SQLite ``SAVEPOINT`` (each inner call opens a
savepoint at the current depth and releases / rolls back on
commit / exception). This is the pattern that lets repositories call
``connection()`` without knowing whether the caller started a
transaction.
"""

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

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection, auto-enrolling in any active transaction.

        If a transaction is active in the current context (thread or
        asyncio task), yields the transaction's connection (the
        transaction manager owns its lifetime). Otherwise opens a new
        connection, commits any pending writes on successful exit, and
        closes it (a raised exception triggers an implicit rollback via
        ``close()``).

        Use this for read-only or single-statement work, and for any
        call that should join an outer ``with transaction():`` block.
        """
        txn = self._active_transaction.get()
        if txn is not None:
            yield txn.connection
            return

        conn = self._connection()
        try:
            yield conn
        except BaseException:
            conn.close()
            raise
        else:
            conn.commit()
            conn.close()

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
