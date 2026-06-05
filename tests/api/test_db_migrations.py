"""Tests for DBMigrations — upgrade, rollback, and SQL file discovery."""

import sqlite3

import pytest

from yadc.api.modules.db_migrations import DBMigrations, _discover_migrations


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
