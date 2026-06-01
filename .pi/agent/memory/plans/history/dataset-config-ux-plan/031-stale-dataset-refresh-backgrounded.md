---
date: 2026-06-01
---
# Stale Dataset Refresh Moved to Background Thread

## Context

After the repository-pattern refactor (history entry 028), `DatasetService.list_datasets`
and `DatasetService.get_dataset` each called `self._refresh_stale_datasets()` before
returning. With the new `with self._db.transaction():` boundary inside
`_apply_disk_scan`, each stale dataset now opens its own short-lived write
transaction (one conn per dataset, instead of the old single-conn-for-the-whole-loop).

When the dataset browser page fires many parallel requests (e.g. list + each
dataset detail), each one triggers a refresh. Two threads can be inside
`_apply_disk_scan` simultaneously, each holding a write transaction. The
second writer waits up to `busy_timeout=5000ms`, then fails with
`database is locked` — visible in logs as
`Failed to refresh dataset id=<N>: database is locked`.

## Decision

Move the stale-dataset refresh off the request path entirely. Schedule it
as a background job in `DatasetService.__init__` via the existing
`JobScheduler`, and have `list_datasets` / `get_dataset` read whatever
state is currently in the index.

To keep the future manual refresh safe (added as a todo to this plan), the
background refresh is wrapped in a `threading.Lock`. Any future caller of
`_refresh_stale_datasets` (manual button, save-as-default side effect, etc.)
serializes with the background job instead of racing for the write lock.

## What changed

### `yadc/api/configuration.py`

Added a new field:

```python
# Dataset index
dataset_refresh_interval_seconds: float = 60.0
```

The interval controls how often the background job re-scans stale datasets.
The staleness cutoff is the same value (datasets scanned within the last
interval are skipped). Default 60s matches the old on-request behavior.

### `yadc/api/services/datasets.py`

- New dependency: optional `JobScheduler` (same pattern as
  `DatasetUploadService`).
- `__init__` creates `self._refresh_lock = threading.Lock()` and, if a
  scheduler was provided, schedules
  `job_scheduler.new_scheduled_job(self._configuration.dataset_refresh_interval_seconds, self._refresh_stale_datasets)`.
- `list_datasets` and `get_dataset` no longer call `_refresh_stale_datasets`.
  They just return repo data. Updated docstrings to point to the background
  job.
- `_refresh_stale_datasets` now acquires `self._refresh_lock` for the
  duration of the loop. `max_age_seconds` defaults to the configured
  refresh interval (was hardcoded `60.0`); tests can pass a smaller value
  to make freshly-registered datasets eligible.

### Tests

The `TestApplyDiskScanOrchestration` fixture in `test_datasets_service.py`
constructs a real `DatasetService` without a `JobScheduler`, so the
background job is not scheduled and the existing tests (which only call
`register` / `rescan_dataset` / `_apply_disk_scan` directly) still pass
without changes. 300 tests pass, no regressions.

## Tradeoffs

- **Latency on external edits**: If a user edits a dataset's config.toml
  externally (e.g. adds a new image path), the change is invisible to the
  API for up to one refresh interval. Mitigated by the inotify watcher
  (still fires for files in already-watched directories) and the planned
  manual refresh button (added as a todo).
- **First-request freshness**: A request immediately after server start
  may see a slightly stale index until the first background pass
  completes. Acceptable; the inotify watcher still picks up any changes
  that happen during the window.
- **No other API surface changes**: `list_datasets` / `get_dataset` are
  now strict reads of the SQLite state. If a caller needs the freshest
  possible view, they should call `rescan_dataset(name)` first.

## Files touched

- `yadc/api/configuration.py` — new `dataset_refresh_interval_seconds` field.
- `yadc/api/services/datasets.py` — background job, lock, removed on-request
  calls.
- `.pi/agent/memory/plans/dataset-config-ux-plan.md` — new todo for the
  manual refresh button; `last_history` bumped to 31.
