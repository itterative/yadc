CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_t REAL NOT NULL DEFAULT (unixepoch())
);
