---
date: 2025-05-30
---
# Phase 5 Backend: Config Revision History

**Context:** Config writes (PATCH/PUT) had no undo — mistakes required manual TOML editing. Needed a revision history system.

**Implementation:**

### DB Migration (step 5)
- `config_history` table: `id` (PK), `dataset_name` (TEXT), `content` (TEXT), `created_t` (REAL)
- Index on `(dataset_name, created_t DESC)` for fast listing

### ConfigHistoryService (`yadc/api/services/config_history.py`)
- `save_snapshot(dataset_name, content)` — inserts row, prunes to 50 entries
- `list_history(dataset_name, limit, before_id)` — cursor-paginated, most recent first
- `get_entry(entry_id)` — single entry lookup
- `delete_entry(entry_id)` — remove a single entry
- Auto-pruning: `DEFAULT_MAX_ENTRIES = 50`, removes oldest on every save

### API Endpoints (`api_configs.py`)
- `GET /configs/<name>/history` — list history entries (paginated via `limit` + `before_id` query params)
- `POST /configs/<name>/history/<int:entry_id>/restore` — restores a snapshot:
  1. Saves current content to history
  2. Writes historical content to disk
  3. Rescans dataset
  4. Deletes the restored entry (avoid clutter)
  5. Returns the restored config (same as GET)

### Integration
- PUT handler: reads current file → saves snapshot → writes new content
- PATCH handler: reads current content → saves snapshot → writes merged content
- Dry-run PATCH does NOT save a snapshot (no write occurs)
- Controller now accepts `ConfigHistoryService` as 4th parameter (DI auto-injects)

### Files
- `yadc/api/modules/db_migrations.py` — step 5 migration
- NEW: `yadc/api/services/config_history.py` — ConfigHistoryService
- `yadc/api/services/__init__.py` — re-export
- `yadc/api/controllers/api_configs.py` — snapshot calls + history endpoints
- `tests/api/test_configs.py` — updated fixture with mock history service

**Checks:** ruff, basedpyright (0 errors), pytest (168 passed) — all clean.
