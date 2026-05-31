from __future__ import annotations

import logging
import sqlite3
import sys

from .logging_factory import LoggingFactory
from .service import Service


class DBMigrations(Service):
    def __init__(self, logging: LoggingFactory) -> None:
        self.current_migration: int = -1
        self._logger: logging.Logger = logging.get_logger(__name__)

    def _run_migration(self, cursor: sqlite3.Cursor, step: int, script: str):
        if self.current_migration >= step:
            return

        try:
            self._logger.debug("Running DB migration step %d", step)

            cursor.executescript(script)

            self.current_migration = step
            cursor.execute("INSERT OR REPLACE INTO properties VALUES (?, ?)", ("migration_step", str(self.current_migration)))

            self._logger.debug("Finished DB migration step %d", step)
        except Exception as e:
            raise Exception(f"failed migration #{step}: {e}") from e

    def run_migrations(self, conn: sqlite3.Connection):
        cursor = conn.cursor()

        try:
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS properties (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

            cursor = cursor.execute("SELECT value FROM properties WHERE key = ?", ("migration_step",))
            row = cursor.fetchone()

            if row is not None:
                self.current_migration = int(row[0])

            self._logger.debug("Current DB migration step is %d", self.current_migration)

            self._run_migration(cursor, step=1, script=self._migration_initial)
            self._run_migration(cursor, step=2, script=self._migration_settings)
            self._run_migration(cursor, step=3, script=self._migration_datasets)
            self._run_migration(cursor, step=4, script=self._migration_source)
            self._run_migration(cursor, step=5, script=self._migration_config_history)
            self._run_migration(cursor, step=6, script=self._migration_config_history_dataset_id)
            # When adding new migrations, add them here with incrementing step numbers.
        except Exception as e:
            conn.rollback()

            self._logger.exception("Caught an exception while running database migrations: %s", e)

            sys.exit(1)
        finally:
            cursor.close()

    # --- Migration scripts ---

    _migration_initial: str = """
        BEGIN IMMEDIATE;

        CREATE TABLE IF NOT EXISTS properties (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        COMMIT;
    """

    _migration_settings: str = """
        BEGIN IMMEDIATE;

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_t REAL NOT NULL DEFAULT (unixepoch())
        );

        COMMIT;
    """

    _migration_datasets: str = """
        BEGIN IMMEDIATE;

        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            config_path TEXT,
            last_scanned_t REAL,
            image_count INTEGER DEFAULT 0,
            created_t REAL NOT NULL DEFAULT (unixepoch()),
            updated_t REAL NOT NULL DEFAULT (unixepoch())
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_datasets_name ON datasets (name);

        CREATE TABLE IF NOT EXISTS dataset_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            file_name TEXT NOT NULL,
            has_caption INTEGER NOT NULL DEFAULT 0,
            has_toml INTEGER NOT NULL DEFAULT 0,
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0,
            draft_names TEXT NOT NULL DEFAULT '',
            last_modified_t REAL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_images_path ON dataset_images (path);
        COMMIT;
    """

    _migration_source: str = """
        BEGIN IMMEDIATE;

        ALTER TABLE datasets ADD COLUMN source TEXT NOT NULL DEFAULT 'import' CHECK(source IN ('import', 'create', 'upload'));

        COMMIT;
    """

    _migration_config_history: str = """
        BEGIN IMMEDIATE;

        CREATE TABLE IF NOT EXISTS config_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            content TEXT NOT NULL,
            created_t REAL NOT NULL DEFAULT (unixepoch())
        );

        CREATE INDEX IF NOT EXISTS idx_config_history_dataset
            ON config_history (dataset_name, created_t DESC);

        COMMIT;
    """

    _migration_config_history_dataset_id: str = """
        BEGIN IMMEDIATE;

        -- SQLite doesn't support adding FK constraints via ALTER TABLE, so we recreate.
        ALTER TABLE config_history RENAME TO config_history_old;

        CREATE TABLE config_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            created_t REAL NOT NULL DEFAULT (unixepoch())
        );

        INSERT INTO config_history (id, dataset_name, dataset_id, content, created_t)
        SELECT
            ch.id,
            ch.dataset_name,
            d.id,
            ch.content,
            ch.created_t
        FROM config_history_old ch
        LEFT JOIN datasets d ON d.name = ch.dataset_name;

        CREATE INDEX IF NOT EXISTS idx_config_history_dataset ON config_history (dataset_name, created_t DESC);
        CREATE INDEX IF NOT EXISTS idx_config_history_dataset_id ON config_history (dataset_id);

        DROP TABLE config_history_old;

        COMMIT;
    """
