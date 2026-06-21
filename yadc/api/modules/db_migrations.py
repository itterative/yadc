"""SQLite database migration system.

Migrations are stored as numbered SQL file pairs (``<NNNN>_<name>_up.sql`` /
``<NNNN>_<name>_down.sql``) in the ``yadc.api.migrations`` package.  The
:class:`DBMigrations` service discovers them at runtime, applies pending upgrades
in order, and supports rolling back to an earlier step using the stored downgrade
SQL.

**Optional Python hooks**: a step may also pair with sibling
``<NNNN>_<name>_<up,down>.py`` files. If present, the runner loads the
``migrate(conn: sqlite3.Connection)`` function and calls it inside the same
``BEGIN IMMEDIATE`` transaction as the SQL step. A Python exception rolls
back the SQL change too (atomicity with the schema change). Most migrations
stay pure SQL — the hook is ``None`` when no sibling ``.py`` exists.

**Drift detection + rollback durability**: at upgrade time, the
runner stores ``(down_py_module_name, down_py_hash, down_py_source)``
in the sibling ``migration_downgrades_py`` table — the canonical
hook source, its module name, and a SHA-256 of the source. On
rollback:

- The stored source is ``exec()``d **in a restricted sandbox**
  (:mod:`yadc.api.modules.sandbox`) and its ``migrate(conn)`` runs
  — exactly like ``down_sql``, the DB row is canonical regardless
  of whether the source package has the hook module. This makes
  Python-hook rollbacks work when the hook **module** is gone (a
  package downgrade that dropped the hook). Note the asymmetry:
  capture (at upgrade time) needs the source file on disk, so this
  covers package downgrades — not frozen executables that ship only
  ``.pyc`` (where capture itself fails). The sandbox limits blast
  radius if a stored row is corrupted (reduced ``__builtins__``, whitelist-only imports).
- ``down_py_hash`` is used for two verification checks:
  - **Stored source rehash** — integrity check on the DB row
    itself (catches tampering of the stored source).
  - **In-package source hash** — drift detection on the source
    package's hook file. Mismatch → warning logged, but the
    rollback proceeds using the stored source (the DB is
    canonical).

The stored source is **trusted Python code that the yadc developer
wrote** — same trust model as the in-package source. The webui is
local-only (not internet-exposed), so an attacker with write access
to the yadc DB also has access to the yadc package; the hash is a
debugging aid and audit-trail tool, not a security boundary.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import re
import sqlite3
from collections.abc import Callable
from importlib.resources import files
from importlib.resources.abc import Traversable

from .logging_factory import LoggingFactory
from .sandbox import StoredHookError, exec_hook
from .service import Service

_MIGRATIONS_PACKAGE = "yadc.api.migrations"
_MIGRATIONS_PATTERN = re.compile(r"^(\d+)_(\w+)_(up|down)\.sql$")
# Optional Python migration hook. The runner looks for a sibling
# ``<step>_<name>_<direction>.py`` file alongside each ``.sql`` file
# and, if present, calls its ``migrate(conn)`` function inside the
# same transaction as the SQL step. Purely additive: most migrations
# stay pure SQL, the hook is ``None`` when no sibling ``.py`` exists.
_PY_HOOKS_PATTERN = re.compile(r"^(\d+)_(\w+)_(up|down)\.py$")


def _capture_hook(module_name: str) -> tuple[str, str]:
    """Return ``(whole-module source, SHA-256 hex)`` for the named hook module.

    Used at upgrade time (to store) and at rollback time (to verify,
    via :func:`_hash_hook_source`).

    Captures the **whole module** rather than just ``def migrate``: the
    function body references module-level imports (``base64``, ``json``,
    …) and the ``from __future__ import annotations`` directive. Without
    that directive the ``sqlite3.Connection`` annotation is evaluated at
    ``def``-time and fails when the source is re-``exec``d in a bare
    namespace on rollback — so a function-only capture crashes on the
    very rollback it was stored for.
    """
    module = importlib.import_module(f"{_MIGRATIONS_PACKAGE}.{module_name}")
    source = inspect.getsource(module)
    return source, hashlib.sha256(source.encode("utf-8")).hexdigest()


def _hash_hook_source(module_name: str) -> str:
    """SHA-256 of the named hook module's whole-module source. See :func:`_capture_hook`."""
    return _capture_hook(module_name)[1]


def _exec_stored_hook(source: str, module_name: str) -> Callable[[sqlite3.Connection], None]:
    """Exec the stored hook source in a restricted sandbox and return its ``migrate(conn)`` callable.

    Thin wrapper over :func:`yadc.api.modules.sandbox.exec_hook` that maps
    sandbox shape errors to :class:`MigrationError`. The sandbox restricts
    ``__builtins__`` and ``__import__`` to limit blast radius if a stored
    row is corrupted — see its module docstring.
    """
    try:
        return exec_hook(source, module_name)
    except StoredHookError as e:
        raise MigrationError(str(e)) from e


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
    - ``migration_downgrades_py`` — drift-detector data for steps with a Python
      migration hook. Stores the hook's module name + the SHA-256 hash of its
      ``migrate(conn)`` source + the source itself so rollback can re-``exec``
      it even when the hook module is no longer importable (package downgrade).
      A row exists only for steps with a hook; absence means "no hook for this
      step". See the module docstring for the threat model.
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

        The tracking tables (``properties``, ``migration_downgrades``,
        ``migration_downgrades_py``) are deliberately bootstrapped here
        rather than created via migration files — the runner needs to
        query them BEFORE any migration runs, so a migration that adds
        their columns can't be the place that introduces them.
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
            -- Drift-detector + rollback durability for optional Python
            -- migration hooks. A row exists only for steps with a Python
            -- hook; absence means "no hook for this step".
            -- ``down_py_source`` is the canonical hook source (run via
            -- ``exec()`` on rollback — same trust model as
            -- ``down_sql``: trusted developer-written code, local DB,
            -- not internet-exposed). ``down_py_hash`` is a SHA-256 of
            -- the source, used for integrity verification on the stored
            -- row + drift detection on the in-package source.
            -- ``ON DELETE CASCADE`` keeps the table in sync with
            -- ``migration_downgrades`` when a step's downgrade row is
            -- removed during rollback.
            CREATE TABLE IF NOT EXISTS migration_downgrades_py (
                step INTEGER PRIMARY KEY REFERENCES migration_downgrades(step) ON DELETE CASCADE,
                down_py_module_name TEXT NOT NULL,
                down_py_hash TEXT NOT NULL,
                down_py_source TEXT NOT NULL
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
            # Optional Python hook runs inside the same transaction so
            # a Python-side exception rolls back the SQL change too.
            up_py = migration.up_py
            if up_py is not None:
                self._logger.debug("Running Python migrate hook for up step %d", migration.step)
                up_py(conn)
            conn.execute("INSERT OR REPLACE INTO properties (key, value) VALUES ('migration_step', ?)", (str(migration.step),))
            if migration.down_sql is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO migration_downgrades (step, down_sql) VALUES (?, ?)",
                    (migration.step, migration.down_sql),
                )
                # Store the down hook's module name + a SHA-256 hash of
                # the source + the source itself so rollback can run
                # even if the source package is gone (same durability
                # as ``down_sql``). Rows in ``migration_downgrades_py``
                # exist ONLY for steps with a hook — absence means
                # "no hook for this step". The FK cascade cleans up
                # the py row when the downgrade row is deleted during
                # rollback.
                down_py_module_name: str | None = migration._down_py_module_name
                if down_py_module_name is not None:
                    down_py_source, down_py_hash = _capture_hook(down_py_module_name)
                    conn.execute(
                        "INSERT OR REPLACE INTO migration_downgrades_py (step, down_py_module_name, down_py_hash, down_py_source) VALUES (?, ?, ?, ?)",
                        (migration.step, down_py_module_name, down_py_hash, down_py_source),
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
                row = conn.execute(
                    "SELECT down_sql FROM migration_downgrades WHERE step = ?",
                    (step,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"Cannot roll back step {step}: no downgrade SQL stored")

                self._logger.debug("Rolling back DB step %d", step)
                statements = _split_statements(row[0])

                # Look up the stored Python hook (module name + hash
                # + source). If a row exists AND ``down_py_source`` is
                # populated, the stored source is canonical — ``exec()``
                # it and run. The in-package source (if importable) is
                # checked for drift as a warning only.
                #
                # Two fallback paths:
                # - ``py_row is None`` (no row at all) — the step has
                #   no Python hook. Skip.
                # - ``py_row`` exists but ``down_py_source`` is NULL
                #   (the row pre-dates the ``down_py_source`` column)
                #   — fall back to the importlib path. This handles
                #   backward compat for rows that were inserted before
                #   the column was added (``_ensure_tables`` adds the
                #   column on existing DBs; rows that pre-date the
                #   addition keep ``down_py_source = NULL``).
                py_row = conn.execute(
                    "SELECT down_py_module_name, down_py_hash, down_py_source FROM migration_downgrades_py WHERE step = ?",
                    (step,),
                ).fetchone()

                conn.execute("BEGIN IMMEDIATE")
                try:
                    for stmt in statements:
                        conn.execute(stmt)
                    if py_row is not None and py_row[2] is not None:
                        down_py_module_name, down_py_hash, down_py_source = py_row
                        # Integrity check on the stored row itself.
                        actual_stored_hash = hashlib.sha256(down_py_source.encode("utf-8")).hexdigest()
                        if actual_stored_hash != down_py_hash:
                            raise MigrationError(
                                f"Stored Python hook for step {step} is corrupted "
                                f"(stored hash {down_py_hash[:12]}\u2026, rehashed "
                                f"{actual_stored_hash[:12]}\u2026). Refusing to roll back."
                            )
                        # Drift check on the in-package source (warning
                        # only — the stored source is canonical).
                        try:
                            package_hash = _hash_hook_source(down_py_module_name)
                            if package_hash != down_py_hash:
                                self._logger.warning(
                                    "Python hook for step %d (module %s) has drifted from the "
                                    "stored source (package hash %s\u2026, stored hash %s\u2026). "
                                    "Using the stored source \u2014 the DB row is canonical.",
                                    step,
                                    down_py_module_name,
                                    package_hash[:12],
                                    down_py_hash[:12],
                                )
                        except ImportError:
                            # In-package source not available — the
                            # case the stored source is meant to cover
                            # (package downgrade that dropped the hook
                            # module). The stored source is what runs.
                            self._logger.debug(
                                "In-package source for step %d (module %s) not available; using stored source.",
                                step,
                                down_py_module_name,
                            )
                        # Run the canonical (stored) source.
                        self._logger.debug("Running stored Python migrate hook for down step %d", step)
                        migrate = _exec_stored_hook(down_py_source, down_py_module_name)
                        migrate(conn)
                    else:
                        # Fallback: import from the package directly.
                        # Used when there's no row at all (no Python
                        # hook for this step) or when the row's
                        # ``down_py_source`` is NULL (the row was
                        # inserted before the column was added to the
                        # schema).
                        try:
                            migration = DBMigrations._find_migration(step)
                        except FileNotFoundError:
                            migration = None
                        if migration is not None:
                            down_py = migration.down_py
                            if down_py is not None:
                                self._logger.debug("Running Python migrate hook for down step %d (no stored source)", step)
                                down_py(conn)
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
    """A single numbered migration step, backed by a pair of SQL files.

    May optionally pair with sibling ``<step>_<name>_<direction>.py``
    files for a Python data hook. The hook is loaded lazily on first
    access (via :attr:`up_py` / :attr:`down_py`) and called inside
    the same transaction as the SQL step.
    """

    def __init__(self, step: int, up_file: Traversable, down_file: Traversable | None = None) -> None:
        self.step: int = step
        self.up_file: Traversable = up_file
        self.down_file: Traversable | None = down_file
        self._up_py_module_name: str | None = None
        self._down_py_module_name: str | None = None
        # Optional pre-loaded callables. Set by tests that build
        # ``_Migration`` objects directly without going through
        # ``importlib``. ``None`` means "no pre-loaded hook" — the
        # property will fall back to ``importlib.import_module`` if a
        # module name is set.
        self._up_py_override: Callable[[sqlite3.Connection], None] | None = None
        self._down_py_override: Callable[[sqlite3.Connection], None] | None = None

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

    @property
    def up_py(self) -> Callable[[sqlite3.Connection], None] | None:
        """Load the optional Python data-migration hook for the up step.

        Returns ``None`` when no sibling ``<step>_<name>_up.py`` was
        discovered alongside the SQL files. ``importlib.import_module``
        is cached internally by Python, so the cost of repeated calls
        is negligible.
        """
        if self._up_py_override is not None:
            return self._up_py_override
        if self._up_py_module_name is None:
            return None
        module = importlib.import_module(f"{_MIGRATIONS_PACKAGE}.{self._up_py_module_name}")
        return module.migrate

    @property
    def down_py(self) -> Callable[[sqlite3.Connection], None] | None:
        """Load the optional Python data-migration hook for the down step. See :attr:`up_py`."""
        if self._down_py_override is not None:
            return self._down_py_override
        if self._down_py_module_name is None:
            return None
        module = importlib.import_module(f"{_MIGRATIONS_PACKAGE}.{self._down_py_module_name}")
        return module.migrate


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
    """Build sorted :class:`_Migration` objects from the migrations package.

    Each migration step consists of a pair of SQL files
    (``<step>_<name>_up.sql`` + ``<step>_<name>_down.sql``) and may
    optionally pair with sibling Python hooks
    (``<step>_<name>_up.py`` + ``<step>_<name>_down.py``) for data
    migrations. SQL files are required (as before); Python hooks are
    purely optional — most migrations stay pure SQL.
    """
    global _MIGRATIONS

    if _MIGRATIONS is not None:
        return _MIGRATIONS

    up_migrations: dict[int, _Migration] = {}
    down_migrations: dict[int, Traversable] = {}
    up_py_hooks: dict[int, str] = {}
    down_py_hooks: dict[int, str] = {}

    for f in files(_MIGRATIONS_PACKAGE).iterdir():
        sql_match = _MIGRATIONS_PATTERN.match(f.name)
        py_match = _PY_HOOKS_PATTERN.match(f.name)

        if sql_match is not None:
            step, direction = int(sql_match.group(1)), sql_match.group(3)
            if direction == "up":
                assert step not in up_migrations, f"duplicate migration for step {step}"
                up_migrations[step] = _Migration(step, up_file=f)
            elif direction == "down":
                assert step not in down_migrations, f"duplicate migration for step {step}"
                down_migrations[step] = f
            else:
                raise MigrationError(f"bad migration: {f.name}")
        elif py_match is not None:
            step, direction = int(py_match.group(1)), py_match.group(3)
            module_name = f.name[:-3]  # strip ".py"
            if direction == "up":
                assert step not in up_py_hooks, f"duplicate python migration for step {step}"
                up_py_hooks[step] = module_name
            elif direction == "down":
                assert step not in down_py_hooks, f"duplicate python migration for step {step}"
                down_py_hooks[step] = module_name
            else:
                raise MigrationError(f"bad python migration: {f.name}")

    for up_migration in up_migrations.values():
        down_migration = down_migrations.get(up_migration.step)
        if down_migration is not None:
            up_migration.down_file = down_migration
        up_migration._up_py_module_name = up_py_hooks.get(up_migration.step)
        up_migration._down_py_module_name = down_py_hooks.get(up_migration.step)

    _MIGRATIONS = list(sorted(up_migrations.values(), key=lambda m: m.step))  # pyright: ignore[reportConstantRedefinition]
    return _MIGRATIONS
