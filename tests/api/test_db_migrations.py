"""Tests for DBMigrations — upgrade, rollback, and SQL file discovery."""

import hashlib
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from yadc.api.modules.db_migrations import (
    DBMigrations,
    MigrationError,
    _capture_hook,
    _discover_migrations,
    _exec_stored_hook,
    _Migration,
)


@pytest.fixture
def migrations(logging_factory):
    return DBMigrations(logging_factory)


@pytest.fixture
def db(tmp_path):
    """Create a temporary SQLite database with WAL mode."""
    path = tmp_path / "test.db"
    conn = sqlite3.connect(str(path))
    conn.execute("pragma journal_mode=wal")
    conn.execute("pragma foreign_keys=on")
    yield conn
    conn.close()


@pytest.fixture
def db_with_tracking(db, migrations):
    """A test DB with the migration tracking tables (``properties`` + ``migration_downgrades``) created.

    Required by tests that call :meth:`DBMigrations._run_upgrade` or
    :meth:`DBMigrations.rollback` directly (bypassing
    :meth:`run_migrations`, which would create them itself).
    """
    migrations._ensure_tables(db)
    migrations._load_current_step(db)
    yield db


# ------------------------------------------------------------------
# Discovery helpers
# ------------------------------------------------------------------


class TestDiscoverSteps:
    def test_discovers_numbered_steps(self):
        steps = [m.step for m in _discover_migrations()]
        assert steps == sorted(steps), "steps should be sorted"
        assert all(isinstance(s, int) for s in steps)

    def test_all_steps_are_positive(self):
        for migration in _discover_migrations():
            assert migration.step > 0

    def test_no_gaps(self):
        steps = [m.step for m in _discover_migrations()]
        if not steps:
            pytest.skip("no migration files found")
        assert steps == list(range(steps[0], steps[-1] + 1)), "migration steps should be contiguous"


# ------------------------------------------------------------------
# Upgrade
# ------------------------------------------------------------------


class TestMigrationFormat:
    """Validate that migration SQL files follow the required format."""

    def test_all_up_migrations_beyond_step_1_have_statements(self):
        for migration in _discover_migrations():
            if migration.step <= 1:
                continue
            stmts = migration.up_statements
            assert len(stmts) > 0, f"step {migration.step} has no executable statements in up migration"

    def test_all_down_migrations_beyond_step_1_have_statements(self):
        for migration in _discover_migrations():
            if migration.step <= 1 or migration.down_file is None:
                continue
            stmts = migration.down_statements
            assert len(stmts) > 0, f"step {migration.step} has no executable statements in down migration"

    def test_statements_no_trailing_semicolons(self):
        for migration in _discover_migrations():
            for stmt in migration.up_statements:
                assert not stmt.endswith(";"), f"step {migration.step} up statement should not end with ';': {stmt!r}"
            for stmt in migration.down_statements:
                assert not stmt.endswith(";"), f"step {migration.step} down statement should not end with ';': {stmt!r}"


class TestRunMigrations:
    def test_fresh_db_reaches_latest_step(self, migrations, db):
        migrations.run_migrations(db)
        last_migration = _discover_migrations()[-1]
        assert migrations.current_migration == last_migration.step

    def test_idempotent(self, migrations, db):
        migrations.run_migrations(db)
        first_step = migrations.current_migration
        # Run again — should be a no-op
        migrations.run_migrations(db)
        assert migrations.current_migration == first_step


# ------------------------------------------------------------------
# Rollback
# ------------------------------------------------------------------


class TestRollback:
    def test_rollback_all(self, migrations, db):
        migrations.run_migrations(db)
        initial = migrations.current_migration

        for step in range(initial - 1, 0, -1):
            migrations.rollback(db, target=step)


# ------------------------------------------------------------------
# Python migration hook
# ------------------------------------------------------------------


class TestPythonHook:
    """The optional Python migrate(conn) hook runs inside the same
    transaction as the SQL step. Most migrations stay pure SQL (no
    sibling ``.py`` file) — the hook machinery is purely additive.

    These tests use an in-memory ``sqlite3.Connection`` plus a
    hand-crafted ``_Migration`` object that points at synthetic SQL
    + Python hook callables — they exercise the runner's hook plumbing
    without depending on a specific migration step's existence.
    """

    def test_python_hook_runs_after_sql_in_same_transaction(self, migrations, db_with_tracking):
        """The up hook is called inside the same transaction as the up SQL."""
        # Create a one-step synthetic migration: a SQL CREATE TABLE +
        # a Python hook that inserts a row using the SAME conn.
        up_file = MagicMock()
        up_file.read_text.return_value = "CREATE TABLE py_hook_test (id INTEGER PRIMARY KEY, n INTEGER);"
        down_file = MagicMock()
        down_file.read_text.return_value = "DROP TABLE IF EXISTS py_hook_test;"

        def fake_up_py(conn):
            conn.execute("INSERT INTO py_hook_test (n) VALUES (42)")

        m = _Migration(step=9999, up_file=up_file, down_file=down_file)
        # Wire the hook by hand (bypass discovery).
        m._up_py_module_name = "fake"  # won't actually be imported
        m._up_py_override = fake_up_py  #

        # Apply.
        migrations.current_migration = 9998  # so the run proceeds
        migrations._run_upgrade(db_with_tracking, m)

        # The hook's INSERT must be visible post-commit.
        rows = db_with_tracking.execute("SELECT n FROM py_hook_test").fetchall()
        assert rows == [(42,)]

    def test_python_hook_exception_rolls_back_sql(self, migrations, db_with_tracking):
        """A Python hook that raises rolls back the SQL change too."""
        up_file = MagicMock()
        up_file.read_text.return_value = "CREATE TABLE py_hook_rollback_test (id INTEGER PRIMARY KEY);"
        down_file = MagicMock()
        down_file.read_text.return_value = "DROP TABLE IF EXISTS py_hook_rollback_test;"

        def explode(conn):
            raise RuntimeError("boom from the hook")

        m = _Migration(step=9998, up_file=up_file, down_file=down_file)
        m._up_py_module_name = "fake"
        m._up_py_override = explode  #

        migrations.current_migration = 9997
        with pytest.raises(RuntimeError, match="boom from the hook"):
            migrations._run_upgrade(db_with_tracking, m)

        # The CREATE TABLE must NOT have survived the rollback.
        row = db_with_tracking.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='py_hook_rollback_test'").fetchone()
        assert row is None

    def test_python_down_hook_runs_during_rollback(self, migrations, db_with_tracking):
        """The stored down hook runs inside the same transaction as the down SQL.

        Seeds a source table, makes the down SQL ``CREATE`` a sink table, and
        stores a hook that copies a row from source to sink. Post-rollback the
        sink row exists only if the hook ran inside the same transaction that
        created the sink table. Exercises the stored-source ``exec`` path
        (``migration_downgrades_py`` row with a populated ``down_py_source``).
        """
        db_with_tracking.execute("CREATE TABLE py_down_source (marker TEXT)")
        db_with_tracking.execute("INSERT INTO py_down_source (marker) VALUES ('seed')")
        db_with_tracking.commit()

        down_sql = "CREATE TABLE py_down_sink (marker TEXT);"
        hook_source = (
            "def migrate(conn):\n"
            "    row = conn.execute('SELECT marker FROM py_down_source').fetchone()\n"
            "    assert row == ('seed',)\n"
            "    conn.execute('INSERT INTO py_down_sink (marker) VALUES (?)', (row[0],))\n"
        )

        db_with_tracking.execute(
            "INSERT OR REPLACE INTO migration_downgrades (step, down_sql) VALUES (?, ?)",
            (9996, down_sql),
        )
        db_with_tracking.execute(
            "INSERT OR REPLACE INTO migration_downgrades_py (step, down_py_module_name, down_py_hash, down_py_source) VALUES (?, ?, ?, ?)",
            (9996, "fake_down", hashlib.sha256(hook_source.encode("utf-8")).hexdigest(), hook_source),
        )
        db_with_tracking.execute(
            "INSERT OR REPLACE INTO properties (key, value) VALUES ('migration_step', ?)",
            ("9996",),
        )
        db_with_tracking.commit()

        migrations.rollback(db_with_tracking, target=9995)

        # The hook ran (copied the seed) inside the same transaction as the
        # down SQL (which created the sink table).
        rows = db_with_tracking.execute("SELECT marker FROM py_down_sink").fetchall()
        assert rows == [("seed",)]

        step = db_with_tracking.execute("SELECT value FROM properties WHERE key = 'migration_step'").fetchone()[0]
        assert int(step) == 9995

    def test_migration_without_py_files_has_no_hook(self):
        """A migration step with only ``.sql`` files returns ``None`` for both hooks."""
        up_file = MagicMock()
        down_file = MagicMock()
        m = _Migration(step=1, up_file=up_file, down_file=down_file)
        # No ``_up_py_module_name`` set → both hooks are ``None``.
        assert m.up_py is None
        assert m.down_py is None

    def test_up_py_lazy_loads_module(self):
        """``up_py`` imports the named module and returns its ``migrate`` attribute."""
        from yadc.api.modules import db_migrations

        up_file = MagicMock()
        m = _Migration(step=1, up_file=up_file)
        m._up_py_module_name = "fake_mod"

        fake_module = MagicMock()
        fake_module.migrate = lambda conn: None

        with patch.object(db_migrations.importlib, "import_module", return_value=fake_module) as imp:
            result = m.up_py

        assert result is fake_module.migrate
        imp.assert_called_once_with("yadc.api.migrations.fake_mod")

    def test_overrides_take_precedence_over_module_import(self):
        """A pre-loaded ``_up_py_override`` is returned without importing."""
        from yadc.api.modules import db_migrations

        up_file = MagicMock()
        m = _Migration(step=1, up_file=up_file)
        m._up_py_module_name = "fake_mod"  # would normally trigger import

        def sentinel(conn: sqlite3.Connection) -> None:
            return None

        m._up_py_override = sentinel

        with patch.object(db_migrations.importlib, "import_module") as imp:
            result = m.up_py

        assert result is sentinel
        imp.assert_not_called()


# ------------------------------------------------------------------
# Python migration hook source capture
# ------------------------------------------------------------------


class TestHookCapture:
    """Guard the whole-module capture path.

    ``_capture_hook`` grabs the whole module (``__future__`` + imports +
    ``def migrate``), not just the function. Function-only capture breaks
    re-``exec`` at rollback time (lost imports + eagerly-evaluated
    annotations). The fixture ``yadc.api.migrations._hook_capture_fixture``
    is a permanent, on-disk module that mirrors a real hook's shape;
    discovery ignores it (its name matches neither migration regex), so it
    costs nothing at runtime. Currently no migration ships a Python hook —
    this is the only thing exercising the capture path.
    """

    def test_capture_hook_returns_whole_module_source(self):
        source, digest = _capture_hook("_hook_capture_fixture")
        # Whole-module capture, not function-only.
        assert "from __future__ import annotations" in source
        assert "import sqlite3" in source
        assert "def migrate" in source
        assert len(digest) == 64

    def test_captured_source_execs_in_sandbox(self):
        source, _ = _capture_hook("_hook_capture_fixture")
        migrate = _exec_stored_hook(source, "_hook_capture_fixture")
        assert callable(migrate)
        migrate(sqlite3.connect(":memory:"))  # runs without error


# ------------------------------------------------------------------
# Hash drift detection on stored hook source
# ------------------------------------------------------------------


class TestHashDriftDetection:
    """The runner stores the down hook's source hash at upgrade time and verifies it on rollback.

    Mismatches raise :class:`MigrationError` so the user gets a loud failure
    if the stored row is corrupted. The hash is a drift detector / integrity
    check, not a tamper-evidence mechanism — see the ``db_migrations`` module
    docstring for the threat model. These tests use synthetic steps (no real
    migration ships a hook) so they don't depend on a specific step number.
    """

    def test_no_py_row_for_steps_without_hook(self, tmp_path, logging_factory):
        """No migration step ships a Python hook → ``migration_downgrades_py`` is empty after a fresh upgrade."""
        db_mig = DBMigrations(logging_factory)
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        conn.execute("pragma foreign_keys=on")
        db_mig.run_migrations(conn)
        rows = conn.execute("SELECT step FROM migration_downgrades_py").fetchall()
        assert rows == [], "no migration step should have a Python-hook row"

    def test_hash_mismatch_raises_on_rollback(self, migrations, db_with_tracking):
        """A stored row whose hash doesn't match the stored source refuses to roll back.

        Integrity check on the DB row itself (catches tampering of the stored
        source). Distinct from the in-package drift check, which only warns.
        """
        source = "def migrate(conn):\n    pass\n"
        db_with_tracking.execute(
            "INSERT OR REPLACE INTO migration_downgrades (step, down_sql) VALUES (?, ?)",
            (9999, "SELECT 1;"),
        )
        db_with_tracking.execute(
            "INSERT OR REPLACE INTO migration_downgrades_py (step, down_py_module_name, down_py_hash, down_py_source) VALUES (?, ?, ?, ?)",
            (9999, "fake_mod", "0" * 64, source),  # deliberately wrong hash
        )
        db_with_tracking.execute(
            "INSERT OR REPLACE INTO properties (key, value) VALUES ('migration_step', ?)",
            ("9999",),
        )
        db_with_tracking.commit()

        with pytest.raises(MigrationError, match="Stored Python hook for step 9999 is corrupted"):
            migrations.rollback(db_with_tracking, target=9998)

        # Rollback aborted before COMMIT → migration_step unchanged.
        step = db_with_tracking.execute("SELECT value FROM properties WHERE key = 'migration_step'").fetchone()[0]
        assert int(step) == 9999

    def test_cascade_deletes_py_row_on_rollback(self, migrations, db_with_tracking):
        """Rolling back a step with a py row removes it via FK ``ON DELETE CASCADE``."""
        source = "def migrate(conn):\n    pass\n"
        correct_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        db_with_tracking.execute(
            "INSERT OR REPLACE INTO migration_downgrades (step, down_sql) VALUES (?, ?)",
            (9999, "SELECT 1;"),
        )
        db_with_tracking.execute(
            "INSERT OR REPLACE INTO migration_downgrades_py (step, down_py_module_name, down_py_hash, down_py_source) VALUES (?, ?, ?, ?)",
            (9999, "fake_mod", correct_hash, source),
        )
        db_with_tracking.execute(
            "INSERT OR REPLACE INTO properties (key, value) VALUES ('migration_step', ?)",
            ("9999",),
        )
        db_with_tracking.commit()

        assert db_with_tracking.execute("SELECT 1 FROM migration_downgrades_py WHERE step = 9999").fetchone() is not None

        migrations.rollback(db_with_tracking, target=9998)

        # FK CASCADE removed the py row when the downgrade row was deleted.
        assert db_with_tracking.execute("SELECT 1 FROM migration_downgrades_py WHERE step = 9999").fetchone() is None
        assert int(db_with_tracking.execute("SELECT value FROM properties WHERE key = 'migration_step'").fetchone()[0]) == 9998

    def test_hash_check_skipped_when_no_py_row(self, migrations, db_with_tracking):
        """A step without a ``migration_downgrades_py`` row skips the hash check (old migration, no hook)."""
        db_with_tracking.execute(
            "INSERT OR REPLACE INTO migration_downgrades (step, down_sql) VALUES (?, ?)",
            (9999, "SELECT 1;"),
        )
        db_with_tracking.execute(
            "INSERT OR REPLACE INTO properties (key, value) VALUES ('migration_step', ?)",
            ("9999",),
        )
        db_with_tracking.commit()

        # No exception — the hash check is skipped because there's no py row.
        migrations.rollback(db_with_tracking, target=9998)

        step = db_with_tracking.execute("SELECT value FROM properties WHERE key = 'migration_step'").fetchone()[0]
        assert int(step) == 9998
