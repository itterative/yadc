---
date: 2025-05-30
---
# Phase 6: TOML Serialization — tomlkit Migration

**Context:** The `toml` library loses comments, formatting, and multiline strings on parse→serialize round-trips. `tomlkit` preserves all of these.

**Scope:** API config editing only (`api_configs.py`). CLI and other services continue using `toml` — deferred.

**Implementation:**

### toml_merge (`dict_utils.py`)
- New `toml_merge(base: TOMLDocument, override: dict) -> TOMLDocument` — tomlkit-aware deep merge
- Uses `copy.deepcopy` (works with tomlkit containers, preserves comments/formatting)
- Recursive `_merge_into(target, override)` helper merges in-place
- Scalars assigned directly; dicts recurse; everything else deep-copied
- Original `deep_merge` unchanged — used by CLI/other services

### api_configs.py swap
- `import toml` → `import tomlkit`
- `toml.loads()` → `tomlkit.loads()` (returns `TOMLDocument` with comments)
- `toml.dumps()` → `tomlkit.dumps()` (preserves formatting)
- PATCH: `deep_merge(parsed, body)` → `toml_merge(parsed, body)` (comments preserved on untouched sections)
- PUT: validates with `tomlkit.loads()`, content passes through verbatim
- Response `parsed` field: `dict(merged_doc)` / `dict(parsed)` to convert tomlkit containers to plain dicts for JSON

### Dependencies
- `tomlkit==0.15.0` added to `pyproject.toml` alongside existing `toml`

### Files
- `pyproject.toml` — added tomlkit dependency
- `yadc/utils/dict_utils.py` — added `toml_merge` and `_merge_into`
- `yadc/api/controllers/api_configs.py` — swapped to tomlkit

**Checks:** ruff, basedpyright (0 errors), pytest (168 passed) — all clean.

**Deferred:** Migrating CLI (`cli_caption.py`, `cli_draft.py`, `cli_export.py`, `cmd/config.py`, `cmd/app.py`) and other API services (`datasets.py`, `captioning.py`, `dataset_upload.py`, `api_export.py`) from `toml` to `tomlkit`.
