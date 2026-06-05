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
