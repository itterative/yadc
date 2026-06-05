CREATE TABLE IF NOT EXISTS config_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL,
    content TEXT NOT NULL,
    created_t REAL NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_config_history_dataset
    ON config_history (dataset_name, created_t DESC);
