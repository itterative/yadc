---
date: 2026-05-31
---
# Relative Paths in Managed Dataset Configs

## Problem

Managed datasets currently store **absolute paths** in their `config.toml`:

```toml
[[dataset]]
path = "/home/user/.local/share/yadc/<name>/images"

[[dataset]]
path = "/home/user/.local/share/yadc/<name>/folders/train"
```

This makes configs **non-portable** — copying `STATE_PATH/<name>/` to another machine or user breaks the paths. The scanner also resolves `entry.path` via `Path(entry.path)` which uses the current working directory, so relative paths would fail if the CWD differs.

## Solution

Store **relative paths** in managed configs and resolve them against `config.toml`'s parent directory at read time.

```toml
[[dataset]]
path = "images"

[[dataset]]
path = "folders/train"
```

### Database impact

`dataset_images.path` stores **absolute paths** (e.g. `/abs/path/to/state/<name>/images/foo.jpg`). The unique index `idx_dataset_images_path` is on `path` alone (global across datasets). This is a pre-existing latent issue but **unaffected by this change** — resolution happens before insertion, so the DB never sees relative paths.

### Changes

1. **`dataset_upload.py`** — Write relative paths:
   - `create_dataset_from_upload()`: `"images"`, `"folders/<name>"`
   - `commit_staged_upload()`: Same, with backward-compatible dedup checking both relative and absolute forms

2. **`datasets.py`** — Resolve at read time:
   - `_scan_dataset()`: If `entry.path` is relative, resolve against `config_file.parent`
   - `_get_dataset_paths()`: Same
   - `import_dataset()`: Keep existing behavior (resolves relatives and writes absolutes back to original file)
   - `create_dataset()`: Keep existing behavior (stores absolute user-provided paths)

3. **`core/dataset_resolver.py`** — Add optional `base_dir` parameter:
   - `resolve_dataset(entries, caption_suffix, base_dir=None)`
   - Resolves relative `entry.path` against `base_dir` before checking `is_dir()`
   - CLI callers (no `base_dir`) unaffected

4. **`api/services/captioning.py`** and **`api/controllers/api_export.py`** — Pass `Path(config_path).parent` as `base_dir` to `resolve_dataset()`

5. **Tests** — Update assertions that check for absolute paths in managed configs
