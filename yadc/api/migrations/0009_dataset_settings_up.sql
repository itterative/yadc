CREATE TABLE IF NOT EXISTS dataset_settings (
    dataset_id INTEGER NOT NULL,
    key        TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    updated_t  REAL    NOT NULL DEFAULT (unixepoch()),
    PRIMARY KEY (dataset_id, key),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dataset_settings_dataset ON dataset_settings (dataset_id);
