---
date: 2026-05-31
---
# Investigation: Managed Dataset Paths and Lifecycle Edge Cases

## Current Architecture

Managed datasets (source="upload") live at `STATE_PATH/<name>/` with this layout:

```
STATE_PATH/<name>/
  config.toml          # Config with absolute paths to images/ and folders/*/
  images/              # Root-level uploaded files (flat)
  folders/             # Top-level folder uploads
    folder_a/
    folder_b/
  .staging/            # Staging dirs for conflict handling (auto-cleaned)
    <uuid>/
```

The config TOML stores **absolute paths** in `[[dataset]]` entries:

```toml
[[dataset]]
path = "/abs/path/to/state/<name>/images"

[[dataset]]
path = "/abs/path/to/state/<name>/folders/folder_a"
```

## What Works Well

- Upload creates the structure correctly
- Append upload with staging + conflict resolution works
- `_scan_dataset()` indexes images from all configured paths
- External file deletions are detected on next scan (images removed from SQLite index)
- `unregister_dataset()` deletes the entire state dir cleanly

## Edge Cases and Gaps

### 1. No API for deleting individual images or folders

**Current state:** There is no `DELETE /datasets/<name>/images/<id>` endpoint. Users cannot remove individual images or folders from a managed dataset through the UI.

**Workaround:** Users must manually delete files from disk and wait for the watcher/refresh to update the index.

**Impact:** High — users will want to curate their datasets.

### 2. Orphaned sidecars when images are deleted

**Current state:** `_scan_dataset()` only tracks image files. Sidecars (`.txt`, `.toml`, `.draft~`, `.history~`) are discovered at caption-read time, not indexed.

**Scenario:**
1. User has `foo.jpg` + `foo.txt` + `foo.toml`
2. Image is deleted externally (or via a future delete API)
3. Next scan removes `foo.jpg` from the SQLite index
4. `foo.txt` and `foo.toml` remain on disk forever — the scanner doesn't know about them
5. These orphaned sidecars accumulate over time

**Impact:** Medium — disk space waste, potential confusion.

### 3. Empty `[[dataset]]` entries in config TOML

**Current state:** When all images in a folder are deleted, the folder path remains in `config.toml` as a `[[dataset]]` entry.

**Scenario:**
1. User uploads `folder_a/img.jpg` → config gets `path = ".../folders/folder_a"`
2. All images in `folder_a/` are deleted
3. Next scan: no images found in that path, index is empty for that dir
4. Config TOML still has the `[[dataset]]` entry
5. Re-uploading to `folder_a/` later works (path deduplicated), but the stale entry was harmless

**Impact:** Low — harmless but clutters config.

### 4. Empty `images/` root directory

**Current state:** Same as #3 — if all root-level images are deleted, the `images/` path stays in config.

**Impact:** Low.

### 5. Staging dir cleanup edge case

**Current state:** `_cleanup_staging_dirs()` iterates all directories under `STATE_PATH/` looking for `.staging/` subdirs.

**Gap:** If the dataset is renamed or deleted, old staging dirs under the old name are never cleaned up (because the old state dir was removed by `unregister_dataset`, which does clean up). But if the state dir persists and `.staging/` has subdirs that are <24h old but the upload was abandoned, they hang around until age-based cleanup.

**Impact:** Low — 24h max lifetime.

### 6. Moving images between folders (user filesystem operation)

**Current state:** `_scan_dataset()` handles this correctly — images appear in the new folder, disappear from the old one. But the config TOML paths don't change.

**Gap:** If a user moves `folder_a/img.jpg` → `folder_b/img.jpg` via filesystem, the next scan updates the index. But if `folder_a/` becomes empty, the config still has a `[[dataset]]` entry for it (see #3).

**Impact:** Low.

### 7. Renaming a managed dataset

**Current state:** There is no rename API. The only way to "rename" is to export/import or recreate.

**Impact:** Medium — users will want to rename datasets.

### 8. Moving a managed dataset to a different machine / backup restore

**Current state:** Config TOML stores **absolute paths**. If the state directory moves (different user, different machine), the absolute paths become invalid.

**Gap:** On `register()` / `rescan_dataset()`, paths are checked with `p.is_dir()`. If the absolute path doesn't exist, the entry is silently skipped. No error is shown.

**Impact:** Medium for portability.

### 9. `_unique_path` collision handling during commit

**Current state:** `_unique_path()` appends `_1`, `_2`, etc. before the extension.

**Gap:** If `foo_1.jpg` already exists (from a previous keep-both), `_unique_path()` returns `foo_1.jpg` which would overwrite it. Actually no — `_unique_path` checks `exists()`, so it would return `foo_2.jpg`. But there's a race condition if two commits happen concurrently.

**Impact:** Very low — concurrent commits to the same dataset are unlikely.

### 10. Config TOML path deduplication is string-based

**Current state:** `existing_paths = {entry.get("path", "") ...}` checks for exact string matches.

**Gap:** If paths differ by trailing slash, symlink, or case (on case-insensitive filesystems), duplicate entries could be created.

**Impact:** Low — paths are generated programmatically and normalized via `.resolve()`.

### 11. Watcher re-registration after config edits

**Current state:** `rescan_dataset()` re-registers the watcher with current paths.

**Gap:** If a user edits the config TOML externally to add/remove paths, the watcher isn't updated until the next `rescan_dataset()` call (which happens every 60s via `_refresh_stale_datasets`, or on filesystem change in an old watched dir).

**Impact:** Low — eventually self-healing.

## Recommendations

### Must-have before merge

1. **Move managed datasets to `STATE_PATH/datasets/<name>/`** — Already on TODO list. Prevents collisions with templates, keys, etc.

### High priority

2. **Add `DELETE /datasets/<name>/images/<id>` endpoint** — Allow users to delete individual images from the UI. Must also delete associated sidecars (`.txt`, `.toml`, `.draft~`, `.history~`).
3. **Add folder deletion** — Either as a separate API or as part of image deletion (deleting the last image in a folder could optionally remove the folder and its config entry).

### Medium priority

4. **Clean up empty `[[dataset]]` entries** — During `rescan_dataset()`, remove config entries whose paths point to empty directories (or non-existent directories).
5. **Clean up orphaned sidecars** — During `rescan_dataset()`, scan for sidecars without matching images and either delete them or log warnings.
6. **Support relative paths in managed dataset configs** — Store paths as relative to `config.toml` (e.g., `./images`, `./folders/train`). This makes configs portable across machines. Resolve to absolute at scan time.

### Low priority / deferred

7. **Rename dataset API** — Copy state dir, update SQLite, delete old.
8. **Race-safe `_unique_path`** — Use a lock or atomic move (not urgent).
