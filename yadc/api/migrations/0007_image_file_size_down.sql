-- SQLite can't DROP COLUMN before 3.35; rebuild dataset_images without
-- file_size (same rename/recreate/copy/drop pattern as 0004_source_down.sql).
-- file_size values are dropped but re-derivable from disk on the next scan.
ALTER TABLE dataset_images RENAME TO dataset_images_old;

CREATE TABLE dataset_images (
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

INSERT INTO dataset_images (id, dataset_id, path, file_name, has_caption, has_toml, width, height, draft_names, last_modified_t)
SELECT id, dataset_id, path, file_name, has_caption, has_toml, width, height, draft_names, last_modified_t FROM dataset_images_old;

DROP TABLE dataset_images_old;

-- Recreate the unique path index so ON CONFLICT(path) in upsert_image keeps working.
CREATE UNIQUE INDEX idx_dataset_images_path ON dataset_images (path);
