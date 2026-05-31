---
date: 2026-05-30
---
# Fix: Path Editing Watcher Re-registration

**Context:** The plan Phase 1 had an open todo: "Path editing for external datasets — verify that changing a dataset entry's path works correctly with the backend's `DatasetWatcherService`." Upon investigation, `rescan_dataset()` re-indexed images from the new path but never updated the filesystem watcher, which continued watching the old directory.

**Problem:**
1. After editing a dataset path via the form, `PATCH /configs/<name>` → `rescan_dataset()` scanned the new directory and updated SQLite, but `DatasetWatcherService._watches` still pointed to the old paths.
2. New files added to the new path wouldn't trigger `DatasetChangedEvent` → no auto-rescan.
3. Conversely, the old (now-unused) directory was still being watched, generating spurious events.
4. If all paths were removed from the config, the old watcher stayed active indefinitely.

**Fix (part 1):** Added watcher re-registration to `DatasetService.rescan_dataset()`:

```python
# After _scan_dataset() + commit:
if config_path:
    config = self._load_config(Path(config_path))
    paths = self._get_dataset_paths(config)
    self._watcher.watch_dataset(name, paths)
```

`watch_dataset()` already handles the unwatch-then-watch dance atomically (locks, unschedules old watches, schedules new ones). Passing an empty `paths` list correctly unschedules all watches without adding new ones.

**Fix (part 2):** `_refresh_stale_datasets()` also re-registers the watcher after scanning. Previously the stale refresh loop updated SQLite but left the filesystem watcher on potentially stale paths when the TOML was edited externally.

**Files touched:** `yadc/api/services/datasets.py`

**Tests:** All 192 tests pass.
