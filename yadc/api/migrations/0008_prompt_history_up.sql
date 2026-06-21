-- Persisted prompt-generator artifacts (``prompt_history``) and their
-- per-image BLOB storage (``prompt_example_images``).
--
-- ``prompt_history`` holds the non-image parts of a saved prompt: mode
-- (generate/refine), intent, focus, and (for refine) the template body.
-- ``template_content`` is NULL for generate mode and non-NULL for refine;
-- the list view derives ``had_template`` from that.
--
-- ``prompt_example_images`` holds the few-shot example images as raw BLOBs
-- (one row per image, ordered by ``ordinal``). The service layer
-- base64-encodes/decodes at the wire boundary, so the DB stores
-- uncompressed bytes (no base64 inflation). ``subject``/``caption`` live
-- in the child row so the example list is fully reconstructable.
-- ``ON DELETE CASCADE`` keeps image rows in sync with their parent entry —
-- deleting an entry cleans up its images automatically, no service work.
CREATE TABLE IF NOT EXISTS prompt_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL CHECK (mode IN ('generate', 'refine')),
    intent TEXT NOT NULL,
    focus TEXT NOT NULL CHECK (focus IN ('system', 'user', 'both')),
    template_content TEXT,
    created_t REAL NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_prompt_history_created
    ON prompt_history (created_t DESC);

CREATE TABLE IF NOT EXISTS prompt_example_images (
    entry_id INTEGER NOT NULL REFERENCES prompt_history(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    subject TEXT NOT NULL,
    caption TEXT NOT NULL,
    mime TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (entry_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_prompt_example_images_entry
    ON prompt_example_images (entry_id);
