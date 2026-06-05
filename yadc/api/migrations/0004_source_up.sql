ALTER TABLE datasets ADD COLUMN source TEXT NOT NULL DEFAULT 'import' CHECK(source IN ('import', 'create', 'upload'));
