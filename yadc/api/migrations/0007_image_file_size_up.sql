-- Per-image byte size. Existing rows default to 0 and are backfilled by the
-- forced re-scan below: resetting last_scanned_t makes the startup background
-- refresh (_refresh_stale_datasets, delay=0) re-walk every dataset, and the
-- scanner's meta-diff treats a 0 -> real-size change as an update.
ALTER TABLE dataset_images ADD COLUMN file_size INTEGER NOT NULL DEFAULT 0;

-- Force every dataset to be re-scanned at startup so file_size is populated
-- immediately for existing databases, rather than after the refresh interval.
UPDATE datasets SET last_scanned_t = NULL;
