"""SQLite database migration system.

Migrations are stored as numbered SQL file pairs (``<NNNN>_<name>_up.sql`` /
``<NNNN>_<name>_down.sql``) in the ``yadc.api.migrations`` package.  The
:class:`DBMigrations` service discovers them at runtime, applies pending upgrades
in order, and supports rolling back to an earlier step using the stored downgrade
SQL.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from importlib.resources import files
from importlib.resources.abc import Traversable

from .logging_factory import LoggingFactory
from .service import Service

_MIGRATIONS_PACKAGE = "yadc.api.migrations"
_MIGRATIONS_PATTERN = re.compile(r"^(\d+)_(\w+)_(up|down)\.sql$")


class MigrationError(Exception):
    """Raised when a database migration fails."""


class DBMigrations(Service):
    """File-based SQLite migration system with rollback support.

    Migrations are discovered at runtime from the ``yadc.api.migrations`` package.  Each
    migration step consists of a pair of SQL files::

        0001_initial_up.sql      -- applied during upgrade
        0001_initial_down.sql    -- applied during rollback

    Steps must be contiguous (1, 2, 3, …) and are always applied in order.

    **Tracking tables** (created automatically):

    - ``properties`` — a general-purpose key/value store.  The migration step is
      tracked via the ``migration_step`` key.  This table predates the migration
      system and is also used by other parts of the application.
    - ``migration_downgrades`` — stores the ``down.sql`` content for every applied
      step so that rollbacks work even if the package source is no longer available
      (e.g. frozen executables).
    """

    def __init__(self, logging: LoggingFactory) -> None:
        self.current_migration: int = -1
        self._logger: logging.Logger = logging.get_logger(__name__)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_migration(step: int) -> _Migration:
        """Find the migration for the given step number."""
        for m in _discover_migrations():
            if m.step == step:
                return m
        raise FileNotFoundError(f"No migration found for step {step}")

    # ------------------------------------------------------------------
    # Running migrations
    # ------------------------------------------------------------------

    def _ensure_tables(self, conn: sqlite3.Connection) -> None:
        """Create the migration tracking tables if they don't already exist.

        This is called before every upgrade or rollback to guarantee the
        tracking infrastructure exists, even on a brand-new database.

        Note: ``properties`` was originally created by migration step 1 in the
        old hardcoded system.  Now that migrations are file-based, the table is
        bootstrapped here instead so that step 1 can remain a no-op placeholder.
        """
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS properties (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS migration_downgrades (
                step INTEGER PRIMARY KEY,
                down_sql TEXT NOT NULL
            );
        """)

    def _load_current_step(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT value FROM properties WHERE key = 'migration_step'").fetchone()
        self.current_migration = int(row[0]) if row is not None else 0
        self._logger.debug("Current DB migration step is %d", self.current_migration)

    def _run_upgrade(self, conn: sqlite3.Connection, migration: "_Migration") -> None:
        if self.current_migration >= migration.step:
            return

        self._logger.debug("Running DB upgrade step %d", migration.step)

        conn.execute("BEGIN IMMEDIATE")
        try:
            for stmt in migration.up_statements:
                conn.execute(stmt)
            conn.execute("INSERT OR REPLACE INTO properties (key, value) VALUES ('migration_step', ?)", (str(migration.step),))
            if migration.down_sql is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO migration_downgrades (step, down_sql) VALUES (?, ?)",
                    (migration.step, migration.down_sql),
                )
        except BaseException:
            conn.rollback()
            raise
        conn.execute("COMMIT")

        self.current_migration = migration.step
        self._logger.debug("Finished DB upgrade step %d", migration.step)

    def run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run all pending upgrades.

        If the database is at a higher step than the latest available migration
        file (e.g. the application was downgraded), any orphaned steps are rolled
        back before applying upgrades.
        """
        try:
            self._ensure_tables(conn)
            self._load_current_step(conn)

            available = _discover_migrations()

            if len(available) == 0:
                return

            latest = max(available, key=lambda m: m.step)

            if self.current_migration > latest.step:
                self._logger.info("DB is at step %d but latest available migration is %d — rolling back", self.current_migration, latest)
                self.rollback(conn, latest.step, _ensure=False)

            for migration in available:
                self._run_upgrade(conn, migration)
        except Exception as e:
            conn.rollback()
            self._logger.exception("Caught an exception while running database migrations: %s", e)
            raise MigrationError(str(e)) from e

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, conn: sqlite3.Connection, target: int, *, _ensure: bool = True) -> None:
        """Roll back migrations from ``current_migration`` down to *target* (exclusive).

        For each step, the stored ``down_sql`` is executed, the downgrade row is
        deleted, and ``current_migration`` is decremented.

        Example: if ``current_migration`` is 4 and ``target`` is 2, steps 4
        and 3 are rolled back (in that order), leaving the DB at step 2.

        Set ``_ensure=False`` when the caller has already called
        ``_ensure_tables`` and ``_load_current_step`` (e.g. :meth:`run_migrations`).
        """
        try:
            if _ensure:
                if self.current_migration < 0:
                    raise RuntimeError("Rollback called without _ensure before loading current migration step")

                self._ensure_tables(conn)
                self._load_current_step(conn)

            for step in range(self.current_migration, target, -1):
                row = conn.execute("SELECT down_sql FROM migration_downgrades WHERE step = ?", (step,)).fetchone()
                if row is None:
                    raise RuntimeError(f"Cannot roll back step {step}: no downgrade SQL stored")

                self._logger.debug("Rolling back DB step %d", step)
                statements = _split_statements(row[0])

                conn.execute("BEGIN IMMEDIATE")
                try:
                    for stmt in statements:
                        conn.execute(stmt)
                    conn.execute("DELETE FROM migration_downgrades WHERE step = ?", (step,))
                    new_step = step - 1
                    conn.execute("INSERT OR REPLACE INTO properties (key, value) VALUES ('migration_step', ?)", (str(new_step),))
                except BaseException:
                    conn.rollback()
                    raise
                conn.execute("COMMIT")

                self.current_migration = new_step
                self._logger.debug("Rolled back DB step %d, now at step %d", step, new_step)
        except Exception as e:
            conn.rollback()
            self._logger.exception("Caught an exception while rolling back database migrations: %s", e)
            raise


class _Migration:
    """A single numbered migration step, backed by a pair of SQL files."""

    def __init__(self, step: int, up_file: Traversable, down_file: Traversable | None = None) -> None:
        self.step: int = step
        self.up_file: Traversable = up_file
        self.down_file: Traversable | None = down_file

    @property
    def up_statements(self) -> list[str]:
        return _split_statements(self.up_file.read_text("utf-8"))

    @property
    def down_statements(self) -> list[str]:
        if self.down_file is None:
            return []
        return _split_statements(self.down_file.read_text("utf-8"))

    @property
    def up_sql(self) -> str:
        return self.up_file.read_text("utf-8")

    @property
    def down_sql(self) -> str | None:
        if self.down_file is None:
            return None
        return self.down_file.read_text("utf-8")


_SQL_COMMENT_RE = re.compile(r"^\s*--")


def _split_statements(sql: str) -> list[str]:
    """Split SQL into individual statements by accumulating lines and yielding at ``;``.

    Lines are accumulated into the current statement.  When a line ends with
    ``;``, the accumulated block is flushed as one statement (with the ``;``
    stripped).  Leading comment-only blocks are skipped.
    """
    statements: list[str] = []
    current: list[str] = []

    for line in sql.splitlines():
        if not current and not line.strip():
            continue

        # Skip comment-only blocks.
        if _SQL_COMMENT_RE.match(line) and not current:
            continue

        current.append(line)
        if not line.rstrip().endswith(";"):
            continue

        block = "\n".join(current)
        current = []
        statements.append(block.rstrip().rstrip(";"))

    if current:
        block = "\n".join(current)
        statements.append(block.rstrip().rstrip(";"))

    return statements


# Pre-computed at import time — the migrations package is read-only data.
_MIGRATIONS: list[_Migration] | None = None


def _discover_migrations() -> list[_Migration]:
    """Build sorted :class:`_Migration` objects from the migrations package."""
    global _MIGRATIONS

    if _MIGRATIONS is not None:
        return _MIGRATIONS

    up_migrations: dict[int, _Migration] = {}
    down_migrations: dict[int, Traversable] = {}

    for f in files(_MIGRATIONS_PACKAGE).iterdir():
        m = _MIGRATIONS_PATTERN.match(f.name)
        if m is None:
            continue

        step, direction = int(m.group(1)), m.group(3)

        if direction == "up":
            assert step not in up_migrations, f"duplicate migration for step {step}"
            up_migrations[step] = _Migration(step, up_file=f)
        elif direction == "down":
            assert step not in down_migrations, f"duplicate migration for step {step}"
            down_migrations[step] = f
        else:
            raise MigrationError(f"bad migration: {f.name}")

    for up_migration in up_migrations.values():
        down_migration = down_migrations.get(up_migration.step)
        if down_migration is not None:
            up_migration.down_file = down_migration

    _MIGRATIONS = list(sorted(up_migrations.values(), key=lambda m: m.step))  # pyright: ignore[reportConstantRedefinition]
    return _MIGRATIONS
