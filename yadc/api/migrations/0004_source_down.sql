-- SQLite does not support DROP COLUMN before 3.35.0.
-- Recreate the table without the source column.
-- Note: this will lose data if other columns have been altered since migration 3.

ALTER TABLE datasets RENAME TO datasets_old;

CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    config_path TEXT,
    last_scanned_t REAL,
    image_count INTEGER DEFAULT 0,
    created_t REAL NOT NULL DEFAULT (unixepoch()),
    updated_t REAL NOT NULL DEFAULT (unixepoch())
);

INSERT INTO datasets SELECT id, name, config_path, last_scanned_t, image_count, created_t, updated_t FROM datasets_old;

DROP TABLE datasets_old;

CREATE UNIQUE INDEX idx_datasets_name ON datasets (name);
