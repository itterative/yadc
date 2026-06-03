---
name: repository-pattern
description: Repository pattern for API services — repos own SQL + data model, services own transactions + business logic. Covers the DBConnectionFactory.connection/transaction contract, the auto-enrollment mechanism, and what belongs in which layer.
category: architecture
priority: 1
keep_updated: true
---

# Repository Pattern for API Services

The web UI backend services (`yadc/api/services/`) split into two kinds of
classes following a **services-provide-functionality, helpers-help-services**
principle:

- **Services** (`DatasetService`, `SettingsService`, `CaptioningService`, …) —
  business logic, file I/O, watcher integration, event handling, **and the
  transaction boundary for multi-statement work**.
- **Repositories** (`DatasetRepository`, `SettingsRepository`, …) — the
  **data model**, **all SQL**, and **single-connection operations**.

This split is enforced by the DI auto-discovery: any class subclassing
`Service` in `yadc.api.services` is auto-discovered and bound as a singleton.
Repositories are services — they're auto-discovered the same way and can be
injected by type.

## Layered design

| Layer | Owns | Examples |
|-------|------|----------|
| **Service** | Watcher integration, event handlers, file I/O (TOML, captions, drafts, history), the `with self._db.transaction():` boundary, multi-method orchestration | `DatasetService.register`, `_apply_disk_scan`, `update_caption` |
| **Repository** | Data model (`DatasetInfo`, `ImageInfo`), all SQL, row → dataclass mapping, single-connection reads and writes | `DatasetRepository.upsert_dataset`, `list_images`, `apply_scan_diff` does NOT belong here |

Two architectural rules from feedback on the refactor:

1. **Models live in the repository, not the service.** `DatasetInfo` and
   `ImageInfo` are defined in `yadc/api/services/dataset_repository.py` and
   re-exported from `datasets.py` so the public service API is unchanged.
   The repo is a leaf in the dependency graph — no service→repo cycle.

2. **The repo never calls `db.transaction()`.** All repo public methods
   use `db.connection()`. The auto-enrollment mechanism (below) means the
   service's `with db.transaction():` block transparently covers repo
   calls.

## The `DBConnectionFactory` contract

`yadc/api/modules/db_connection_factory.py` provides two connection
acquisition methods, both implemented as context managers (since the
refactor that landed the `connection()` context manager on 2026-06-01):

```python
@contextmanager
def connection(self) -> Iterator[sqlite3.Connection]:
    """Yield a connection, auto-enrolling in any active transaction.

    If a transaction is active in the current context (thread or
    asyncio task), yields the transaction's connection (the
    transaction manager owns its lifetime). Otherwise opens a new
    connection, commits any pending writes on successful exit, and
    closes it (a raised exception triggers an implicit rollback via
    ``close()``).
    """

@contextmanager
def transaction(self) -> Iterator[sqlite3.Connection]:
    """Context manager that yields a connection within a transaction.

    Nesting is supported: inner calls create SQLite savepoints.
    The connection is also returned by ``connection()`` while the
    transaction is active, so existing callers are automatically
    enrolled without changes.
    """
```

Key invariants:

- **`connection()` auto-enrolls in any active transaction.** If a service
  is inside `with db.transaction():` and calls a repo method that uses
  `db.connection()`, the repo's work joins the same transaction (same
  conn, no separate BEGIN).
- **`connection()` commits pending writes on successful exit** (and
  closes on exit; close rolls back on exception). The Python `sqlite3`
  default is to roll back uncommitted writes on close, so the explicit
  `commit()` in the `else` branch is what makes the standalone case
  atomic on its own.
- **`transaction()` closes the conn itself** (in the outermost call's
  `txn.close()`). When a repo's `connection()` yields a transactional
  conn, the repo's context-manager exit doesn't re-close it (the txn
  owns the lifetime; the context manager's `if txn is not None: yield;
  return` short-circuits before any close logic).
- **Nesting is supported** via SQLite savepoints — `with transaction()`
  inside a `with transaction():` block opens a savepoint, not a fresh
  transaction.

## What belongs in the repo

Every repo public method:

- Takes only application-level parameters (no `conn`).
- Internally uses `with self._db.connection() as conn:` for its SQL.
- Returns application-level data: dataclass instances, primitive
  tuples, bools, counts.
- Inlines row → dataclass construction at the call site. Don't extract a
  `_row_to_X` helper "for DRY" — readers want to see the column list
  next to the SQL.

The repo's public API should be **single-statement, single-table-shaped**.
Multi-statement work is composed at the service level by wrapping several
repo calls in a transaction.

## What belongs in the service

The service is the **transaction boundary**. Multi-step work is:

```python
def _apply_disk_scan(self, dataset_id, config_path, config=None):
    """Walk the disk for a dataset and reconcile the index in one transaction."""
    disk_images = self._scan_disk(config_path, config=config)
    with self._db.transaction():
        existing_by_path = self._repo.list_image_paths(dataset_id)
        for path, meta in disk_images.items():
            self._repo.upsert_image(
                dataset_id=dataset_id, path=path, file_name=meta["file_name"],
                has_caption=meta["has_caption"], has_toml=meta["has_toml"],
                width=meta["width"], height=meta["height"],
                draft_names=meta["draft_names"], last_modified_t=meta["last_modified_t"],
            )
        for path, img_id in existing_by_path.items():
            if path not in disk_images:
                self._repo.delete_image(img_id)
        self._repo.update_dataset_stats(dataset_id, len(disk_images))
```

The disk walk (`_scan_disk`) is **service-level** — it's file I/O, not
SQL, and shouldn't hold the transaction open. Do the walk outside the
`with transaction():` block, then enter the transaction only for the
SQL part.

The service also owns:

- Watcher integration (start/stop watching, expect_file_change)
- Event handlers (`@event_handler(DatasetChangedEvent)`)
- File I/O that doesn't touch the DB (TOML parsing, caption files,
  draft files, history files)
- Business policy (e.g. when to rescan, what counts as a "stale"
  dataset)

## Background scans

Long-running disk scans (`_apply_disk_scan` and friends) are
**scheduled as a background job** in `DatasetService.__init__` via the
optional `JobScheduler` dependency, not triggered on every API
request. Doing the scan on each `list_datasets` / `get_dataset` call
made the API vulnerable to "database is locked" errors under parallel
load — the SQLite write lock is global, and concurrent scans would
exceed `busy_timeout` (5s) on a slow dataset.

Pattern:

```python
def __init__(self, ..., job_scheduler: JobScheduler | None = None):
    ...
    self._refresh_lock: threading.Lock = threading.Lock()
    if job_scheduler is not None:
        job_scheduler.new_scheduled_job(
            self._configuration.dataset_refresh_interval_seconds,
            self._refresh_stale_datasets,
        )

def _refresh_stale_datasets(self, max_age_seconds: float | None = None):
    """Rescan stale datasets. Held under _refresh_lock so a future
    manual refresh button can't race the background job."""
    with self._refresh_lock:
        ...
        # Dispatches DatasetChangedEvent for each dataset that had changes.
```

`_apply_disk_scan` returns `True` when rows were actually upserted or
deleted (i.e. the index changed). Both `_refresh_stale_datasets` and
`rescan_dataset` use this return value to dispatch a
`DatasetChangedEvent` only when the scan found real changes — avoiding
spurious SSE notifications to the frontend.
```

The `JobScheduler` is optional so tests can construct the service
without it. The lock serializes the background job with any future
caller (manual refresh button, save-as-default side effect) — callers
block instead of failing with "database is locked".

`list_datasets` / `get_dataset` are now strict reads of the index
state. Inotify events still drive live updates via the watcher; the
stale-refresh job is a fallback for changes the watcher can't see
(e.g. external edits to a dataset's `config.toml` that add new image
paths, where no inotify event fires).

## What the service no longer does

After the refactor:

- **Doesn't open DB connections for its own SQL.** SQL is the repo's
  job. The service may still need a `DBConnectionFactory` for the
  `with self._db.transaction():` boundary, but the SQL itself lives in
  the repo.
- **Doesn't have `try/finally conn.close()` boilerplate.** The
  `connection()` context manager handles it (or the transaction
  manager does, for the txn case).
- **Doesn't have its own dataclasses for SQL-shaped data.** Re-import
  them from the repo (or use the re-exports in `yadc/api/services/__init__.py`).
  Service-level result types like `ImagePage` (paginated result) and
  `HistoryEntry` (history snapshot from a file) stay in the service —
  they're not SQL shapes.

## Test patterns

The test conftest in `tests/api/conftest.py` provides:

- `test_configuration` — a `Configuration` populated with `tmp_path`
  for state/cache/config directories and a temp-file `db_path`.
- `logging_factory` — a `LoggingFactory` wired to the test config.
- `db_connection_factory` — a real `DBConnectionFactory` with the
  migrations applied. Tests get a fresh temp-file DB per test.

Repository tests follow these rules:

- **Use the real factory** — the schema comes from the existing
  migrations. Don't `CREATE TABLE` in the test; the factory does it.
- **Build state by calling the repo's public methods.** No raw SQL
  `INSERT`s in the test. `_insert_dataset(conn, ...)` helper functions
  are an anti-pattern.
- **Assert on the repo's public methods.** Calling `repo.upsert_dataset(...)`
  and then `repo.get_dataset(...)` to verify the round-trip is the
  preferred pattern.
- **For failure / atomicity tests**, use the service (which is where
  the transaction boundary lives now). Drop a table mid-transaction
  from the factory fixture, call the service's multi-statement
  method, assert the prior state is intact.

## Anti-patterns

These came up in code review and are worth flagging:

- **Public methods that take a `conn` parameter.** Use the
  `connection()` context manager inside the method. Repos that take
  `conn` leak the transaction boundary to the caller.
- **Private `_in_txn` helpers** (`_upsert_image_in_txn`,
  `_upsert_dataset_in_txn`, …). They exist because the public method
  is the only caller — inline the SQL in the public method.
- **A `apply_scan_diff` (or similar high-level multi-statement)
  method on the repo.** The service owns the orchestration. Drop it
  and have the service compose granular repo calls inside a
  `with db.transaction():` block.
- **A row-to-dataclass helper** like `_dataset_info_from_row`. Each
  call site should show the column list next to the construction so
  readers can match columns to SQL.
- **Schemas duplicated in test fixtures.** Use the real factory +
  migrations.

## Where the patterns live in the codebase

- `yadc/api/modules/db_connection_factory.py` — the
  `connection()`/`transaction()` contract.
- `yadc/api/services/dataset_repository.py` — example repo. Owns
  `DatasetInfo`, `ImageInfo`, and the SQL for `datasets` +
  `dataset_images`.
- `yadc/api/services/datasets.py` — example service. The
  `_apply_disk_scan` and `register` methods show the
  service-orchestrates-multi-statement pattern.
- `yadc/api/services/settings.py` + `settings_repository.py` —
  smaller example; same pattern, no multi-statement orchestration
  needed.
- `tests/api/conftest.py` — shared fixtures.
- `tests/api/test_dataset_repository.py` and
  `tests/api/test_settings_repository.py` — repo tests using the real
  factory.
- `tests/api/test_datasets_service.py` — the
  `TestApplyDiskScanOrchestration` class shows the
  service-orchestrates-transaction test pattern.

## Related plans and history

- The original proposal is in
  `plans/archive/repository-pattern-proposal.md` (status: in-progress).
- The history entry for the cycle fix is in
  `plans/history/dataset-config-ux-plan/029-managed-datasets-cycle-fix.md`
  (the cycle that motivated extracting `managed_paths.py`, and the
  precursor to this refactor).
