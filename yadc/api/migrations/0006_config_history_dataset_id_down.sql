-- Reverse of step 6: recreate config_history without dataset_id column.
ALTER TABLE config_history RENAME TO config_history_old;

CREATE TABLE config_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL,
    content TEXT NOT NULL,
    created_t REAL NOT NULL DEFAULT (unixepoch())
);

INSERT INTO config_history (id, dataset_name, content, created_t)
SELECT id, dataset_name, content, created_t
FROM config_history_old;

CREATE INDEX IF NOT EXISTS idx_config_history_dataset ON config_history (dataset_name, created_t DESC);

DROP TABLE config_history_old;
