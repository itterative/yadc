-- Reverse of 0008: drop the child image table before the parent it
-- references. SQLite doesn't enforce FK checks on DDL drops, but this
-- order documents the dependency direction.
DROP INDEX IF EXISTS idx_prompt_example_images_entry;
DROP TABLE IF EXISTS prompt_example_images;
DROP INDEX IF EXISTS idx_prompt_history_created;
DROP TABLE IF EXISTS prompt_history;
