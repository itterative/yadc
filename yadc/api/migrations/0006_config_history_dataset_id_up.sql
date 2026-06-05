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
