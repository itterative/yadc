---
name: dataset-watcher
description: Filesystem watcher for dataset directories — inotify via watchdog, debouncing, expected-change tracking (_expected_sources + _expected_files), event dispatch, and frontend suppression.
category: architecture
---

# Dataset Watcher

## Overview

`DatasetWatcherService` (in `yadc/api/modules/dataset_watcher.py`) monitors dataset image directories for filesystem changes using **watchdog** (inotify on Linux). When changes are detected, it debounces them per-dataset and dispatches `DatasetChangedEvent` through the `EventDispatcher`. The frontend uses these events to auto-refresh dataset image lists.

## Key Concepts

### Watched Extensions

Only files with these extensions trigger events:

| Category | Extensions |
|----------|------------|
| Images | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff`, `.tif`, `.ico` |
| Sidecars | `.txt` (captions), `.toml` (metadata), `.history~`, `.draft~` |

### Watchdog Events Handled

`_DirEventHandler` listens for `on_created`, `on_deleted`, `on_closed` (for writes), and `on_moved`. Only non-directory events for watched extensions are forwarded.

## Architecture

```
Filesystem change
  → _DirEventHandler.on_created/on_deleted/on_closed/on_moved
    → _on_fs_change(dataset_name, file_path)
      → Check _expected_files (exact path match?)
      → Check _expected_patterns (fnmatch glob match?)
      → Update _unexpected_changes flag
      → Debounce: cancel old timer, start new one (capture job_id from _expected_sources)
        → _dispatch_change(dataset_name, job_id)
          → DatasetChangedEvent(dataset_name, job_id)
            → DatasetService._on_dataset_changed
              → If job_id (self-originated): skip rescan (DB already updated per-image)
              → If no job_id (external): rescan_dataset
            → Frontend SSE: events.ts listener
```

## Change Registration

Three mechanisms track "expected" changes to suppress self-originated events:

### 1. `_expected_files` — File-level (webui edits + captioning)

- **Set by**: `expect_file_change(dataset_name, file_path, *, source=SELF_JOB_ID)` — called **before** writing/deleting a file
- **Currently used by**: `DatasetService` (webui caption saves, extras updates, history restores, image deletes) and `AsyncCaptionJob._acaption_one()` (captioning writes)
- **Storage**: `dict[str, deque[ExpectedFileEntry]]` — bounded deque per dataset (max 256 entries). Each `ExpectedFileEntry` is a NamedTuple with `path`, `registered_at`, and `source`.
- **`source`**: A client identifier that flows through to the dispatched `DatasetChangedEvent.job_id`. Frontend passes `"ui:<tab-uuid>"`; backend-only callers use the default `SELF_JOB_ID` ("self").
- **Matching**: In `_on_fs_change`, each event checks if the file path matches a registered entry **within TTL** (default 1.0s). Entries are NOT consumed on match — a single write can produce multiple inotify events.
- **Cleanup**: TTL expiry (entries older than `watcher_expected_file_ttl` are ignored during matching). Also cleared on `_unwatch_dataset_locked` and `_dispatch_change`.

### 2. `_expected_patterns` — Pattern-level (bulk deletions)

- **Set by**: `expect_pattern_change(dataset_name, pattern, *, source=SELF_JOB_ID)` — called **before** bulk deletions where enumerating every file is impractical
- **Currently used by**: `DatasetService._delete_file_with_sidecars()` (draft files: `STEM.*.draft~`) and `DatasetService.delete_items()` (folder deletions: `folder/*`)
- **Storage**: `dict[str, deque[ExpectedPatternEntry]]` — same bounded deque as exact entries. Each `ExpectedPatternEntry` is a NamedTuple with `pattern` (an `fnmatch` glob), `registered_at`, and `source`.
- **Matching**: In `_on_fs_change`, after exact-path matching fails, each event checks `fnmatch.fnmatch(file_path, pattern)` **within TTL**. Also NOT consumed on match.
- **Cleanup**: Same TTL and clear-on-dispatch/unwatch semantics as `_expected_files`.

### 3. `_expected_sources` — Dataset-level (captioning jobs)

- **Set by**: `expect_changes(dataset_name, job_id)` — called at captioning job start
- **Cleared by**: `clear_expected_changes_for_job(dataset_name, job_id)` — via a **5-second delayed async cleanup** in `CaptioningService._cleanup_async()` after job completion
- **Purpose**: Tags `DatasetChangedEvent` with the captioning `job_id` so the originating frontend can suppress its own job's rescans while other clients still see the events
- **No longer critical**: File-level `expect_file_change()` calls in `_acaption_one()` provide the primary suppression mechanism. The timer still exists for job_id tag cleanup, but mistimed clears only affect the job_id metadata on events, not suppression correctness.

### How they interact in `_dispatch_change`

| `_unexpected_changes` | `_expected_sources` (job_id) | `_expected_files` + `_expected_patterns` source | Result `job_id` on event | Meaning |
|-----------------------|------------------------------|-------------------------------------------------|--------------------------|---------|
| `False` | `None` | `"ui:abc"` | `"ui:abc"` | All expected from one tab — only that tab suppresses |
| `False` | `None` | mixed / default | `"self"` | Mixed sources — all tabs suppress (legacy fallback) |
| `False` | `"abc123"` | (any) | `"abc123"` | Captioning job_id takes priority |
| `True` | `None` | (any) | `None` | External/unexpected change — all clients refresh |
| `True` | `"abc123"` | (any) | `"abc123"` | Mixed: still tagged with captioning job_id |

## Debouncing

Each `_on_fs_change` call cancels the previous debounce timer and starts a new one (default 1.0s). This coalesces rapid bursts of filesystem events into a single `DatasetChangedEvent`. The `job_id` is snapshotted from `_expected_sources` at schedule time so a later cleanup can't wipe it before dispatch.

## Watch Lifecycle

| Operation | Method | Notes |
|-----------|--------|-------|
| Register | `watch_dataset(name, paths)` | Diffs paths against current watches — short-circuits if unchanged. Preserves `_expected_sources` across re-registration (for stale refresh during captioning). |
| Unregister | `unwatch_dataset(name)` | Clears all state: watches, timers, expected files/sources, unexpected flags. |
| Tag captioning | `expect_changes(name, job_id)` | Set at job start, cleared 5s after job end. |
| Tag file write | `expect_file_change(name, path, *, source)` | Set before each file write/delete, TTL-based expiry. `source` identifies the originating client. |
| Tag pattern | `expect_pattern_change(name, pattern, *, source)` | Set before bulk deletions (drafts, folders). `pattern` is an `fnmatch` glob. Same TTL and deque limit. |

### Callers of `watch_dataset`

| Caller | When |
|--------|------|
| `DatasetService.register_dataset()` | After initial scan |
| `DatasetService.rescan_dataset()` | After re-scan (picks up path changes) |
| `DatasetService._refresh_stale_datasets()` | Periodic refresh of stale datasets |
| `DatasetService._watch_existing_datasets()` | At startup |
| `DatasetService._on_dataset_changed()` | Auto-rescan triggered by watcher event (cycle) |

### Backend Event Handling

`DatasetService._on_dataset_changed` decides whether to rescan based on the event's `job_id`:

| `job_id` | Source | Action |
|----------|--------|--------|
| `None` / `""` | External change | Full `rescan_dataset` |
| `"abc123"` | Captioning job | Skip (per-image `refresh_image_index` already handled it) |
| `"self"` / `"ui:abc"` | Webui edit (any tab) | Skip (API endpoint already updated the DB) |

A final `rescan_dataset` runs in `CaptioningService._cleanup_async()` (5s after job ends) to catch any external changes that occurred during captioning.

### Cycle: External Change → Rescan → Rewatch

```
FS change (external) → _on_fs_change → debounce → _dispatch_change → DatasetChangedEvent(job_id=None)
  → DatasetService._on_dataset_changed → rescan_dataset → watch_dataset
```

Self-originated changes (job_id set) short-circuit at `_on_dataset_changed`. The path-diff in `watch_dataset` prevents redundant unwatch→rewatch when paths haven't changed.

## Frontend Suppression

In `events.ts`, the `dataset_changed` SSE listener:

1. If `job_id` matches an **active job ID** registered by this client → suppress (own captioning job)
2. If `job_id` matches this tab's **`clientId`** (`"ui:<uuid>"`) → suppress (own webui edit)
3. If `job_id === "self"` → suppress (legacy, older backend versions)
4. Otherwise → add to `_pendingDatasetChanges` → show refresh toast

Each frontend tab generates a unique `clientId` on module load (stored in `events.ts`, format `"ui:<random>"`). This ID is sent as a `?source=` query parameter with API requests that cause expected file changes (caption save, extras update, history restore, delete). The backend tags the `DatasetChangedEvent` with this source, allowing only the originating tab to suppress it.

## Files

| File | Role |
|------|------|
| `yadc/api/modules/dataset_watcher.py` | `DatasetWatcherService` — watcher, debouncing, expected-change tracking |
| `yadc/api/services/captioning.py` | `CaptioningService` — sets `expect_changes` at job start, clears on completion |
| `yadc/api/services/datasets.py` | `DatasetService` — calls `watch_dataset`, `expect_file_change` for webui edits, handles `DatasetChangedEvent` |
| `yadc/api/configuration.py` | `watcher_debounce_seconds`, `watcher_expected_file_max`, `watcher_expected_file_ttl` |
| `yadc/webui/src/lib/stores/events.ts` | Frontend SSE listener — per-tab clientId generation, job_id-based suppression |
| `yadc/webui/src/lib/stores/datasetImages.ts` | Frontend API helpers — sends `?source=clientId` with mutating requests |

## Known Issues / Future Work

- **Duplicate inotify events**: Some file writes produce multiple events despite `on_closed` (`IN_CLOSE_WRITE`). The TTL-based buffer mitigates but doesn't eliminate duplicates. May need per-path deduplication in `_on_fs_change`.
