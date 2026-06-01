---
status: in-progress
priority: 2
---
# Repository Pattern for API Services

## Status

**In progress as of 2026-06-01** — rebased onto `feature/svelte-frontend`
which already merged the `DBConnectionFactory.transaction()` infrastructure.
Originally proposed for `main`, but the user re-prioritized this work
over the dataset-config-ux-plan work after the rebase — same intent
(cross-cutting refactor on the API side), now happening on the feature
branch.

## What changed since the original proposal

The simplified `transaction()` context manager in the original proposal was
**superseded by a more capable implementation** in
`feature/svelte-frontend` (commit `15f6292 feat(api): add transaction() with
nested savepoints, synchronous DB init`):

- **`_Transaction` helper class** manages a single connection with depth
  tracking and `BEGIN` / `SAVEPOINT sp_N` / `RELEASE` / `ROLLBACK TO`
  switching.
- **Nested transactions use savepoints** — inner `with db.transaction() as
  conn:` blocks create savepoints inside the outer transaction, so
  multi-layer service code can be composed freely without coordination.
- **`_active_transaction` is a `ContextVar`** (not a `threading.local`),
  so the transaction state flows through both threads and asyncio tasks.
- **Bare `connection()` calls auto-enroll in the active transaction**:
  `connection()` checks the `ContextVar` and returns the transaction's
  connection if one is open. This means existing code (and the refactor's
  intermediate states) keep working without manual `with transaction()`
  wrapping — every call to `connection()` inside a `with transaction()`
  block silently joins it.
- **Synchronous DB init** — `_init_db` no longer runs in a background
  thread; `connection()` (and `transaction()`) wait synchronously.

The repository pattern still applies, but the refactor now needs to be
clearer about when to use `transaction()` vs `connection()`. The
auto-enrollment is a feature, not a bug, but the **service** should still
own the boundary explicitly so the intent is visible at the call site.

## Why still refactor

`yadc/api/services/datasets.py` (1086 lines) and
`yadc/api/services/settings.py` (~75 lines) both contain raw SQL inline in
service methods. The new `transaction()` API doesn't change that — the
boilerplate is the same, the SQL is still scattered.

Specific issues that remain:

- SQL is scattered through service files — hard to grep, hard to test, hard
  to optimize in one place.
- ~25 copies of the try/finally-close pattern in `datasets.py`, 4 in
  `settings.py`.
- `_scan_dataset` still takes `conn` as a hidden parameter passed through
  the service layer — works, but a repo's `upsert_image` would be
  self-documenting.
- Tests mock the connection factory and the SQL — better to test SQL
  against an in-memory DB in repo unit tests, and test the service against
  a fake repo.

## Design (updated for the new `transaction()` API)

### 1. Repositories as `Service` subclasses

Auto-discovered, stateless, take `db: DBConnectionFactory` (+ optional
`logging`). Methods take `conn` as their first parameter — repos do not
manage connections themselves, but they are free to call
`self._db.connection()` for read-only convenience methods where the
auto-enrollment behaviour is fine.

- `yadc/api/services/dataset_repository.py` (`DatasetRepository`)
- `yadc/api/services/settings_repository.py` (`SettingsRepository`)

### 2. Service code uses explicit transaction boundary

```python
# Read (no transaction needed — bare connection auto-enrolls if a
# transaction happens to be open, otherwise it's a fresh short-lived conn):
with self._db.connection() as conn:
    return self._repo.get(conn, name)

# Single-statement write:
with self._db.transaction() as conn:
    self._repo.upsert(conn, ...)

# Bulk write — preserves the _scan_dataset pattern:
with self._db.transaction() as conn:
    for stale in self._repo.list_stale(conn, cutoff):
        self._repo.upsert_image(conn, stale.id, ...)
```

The `with db.connection() as conn:` form is preferred over bare
`conn = self._db.connection()` so the conn is always closed — the
auto-enrollment behaviour does not change connection lifetime (a fresh
conn is still created and closed per call when no transaction is active).

### 3. Service responsibilities after the refactor

- Watcher integration (`DatasetWatcherService`)
- Event handlers (`@event_handler(DatasetChangedEvent)`)
- File-system ↔ DB coordination (caption files, draft files, history files)
- Multi-step orchestration (the rescan loop, the commit flow)
- The `transaction()` boundary

### 4. Repository responsibilities

- All SQL (schema names, query text, parameter binding)
- Row → dataclass conversion (the SQL-tuple-to-DatasetInfo mapping)
- Nothing else

### 5. Tests

- Add focused repo unit tests using an in-memory SQLite DB.
- Existing service tests: prefer real repos over mocks. The repo's
  `connection()` parameter can be a fixture-provided in-memory conn; the
  service's `_repo` attribute can be the real repo pointing at the same
  in-memory DB.
- `DBConnectionFactory` is now auto-init (synchronous), so test fixtures
  no longer need to wait for the background init thread.

## What this does NOT change

- The DI auto-discovery mechanism still works (repos are services).
- No new types or Pydantic models needed (existing dataclasses — `DatasetInfo`,
  `ImageInfo`, `HistoryEntry` — become the return types of repo methods).
- The on-disk schema is unchanged.
- The watcher, event dispatcher, and SSE pipeline are untouched.
- The `transaction()` API is untouched (it's a feature the repos build
  on, not something the repos own).

## Implementation order

1. **`SettingsRepository` first** (4 methods, ~75 LOC). Validates the
   pattern at small scale.
2. **`DatasetRepository` next** (the big one). 30+ methods, ~1086 lines
   to split. Done in two passes:
   - Pass 1: read methods (list/get, paginated queries) — the easy ones
   - Pass 2: write methods (scan, register, update caption/extras/history)
3. Service tests: switch from mocking the connection factory to using
   real repos with an in-memory SQLite DB.

## Estimated impact

- `datasets.py`: 1086 → ~650 lines (coordination only)
- `datasets_repository.py`: new, ~350 lines (all SQL)
- `settings.py`: ~75 → ~30 lines
- `settings_repository.py`: new, ~60 lines
- `db_connection_factory.py`: unchanged (already has the new API)
- `dataset_repository` tests: 0 → ~10-15 tests
- `settings_repository` tests: 0 → ~3-4 tests
- Service tests: existing tests should still pass with fixture changes

## Patterns explicitly rejected

- **Repos own the connection**: hides the transaction boundary inside the
  data layer. Services should own the boundary.
- **Unit-of-work pattern**: overkill for two services of this size. The
  `with db.transaction() as conn` block is the unit of work.
- **Repository returns ORM-style aggregates**: keeps the existing dataclass
  shapes (small, focused, already used by the controllers and WebUI).
- **Leveraging auto-enrollment to skip explicit `transaction()`**: the
  auto-enrollment in `connection()` is a useful backstop, but the service
  should be explicit about transactions for readability. A reader of the
  service should not have to know that `connection()` may or may not
  return a transactional conn.

## Open questions

- Should `DatasetRepository` and `SettingsRepository` be exposed via
  `__all__` in `services/__init__.py`, or kept as internals? My default
  would be keep them internal (controllers should still go through the
  services). Resolved: internal only.
- Test fixture strategy: in-memory SQLite DB per-test, with the migrations
  applied, and the repos wired against it. Avoids the temp-file dance
  while still keeping tests fast.
- Naming: `DatasetRepository` vs `DatasetsRepository`? The service is
  singular `DatasetService` because there's "one" service even though it
  operates on many datasets. The repo operates on the same collection
  shape (rows in `datasets` and `dataset_images` tables). Singular is
  fine.

## Related

- The dataset-config-ux-plan (this feature branch) is the wrong home for
  the *original* plan (cross-cutting architectural refactor, not part of
  the UX work's scope). The user redirected on 2026-06-01 to do this on
  `main`; then re-prioritized after the rebase onto `feature/svelte-frontend`
  brought the `transaction()` infrastructure in. So it lives here for
  now, but is intentionally independent of the dataset-config-ux-plan
  history entries (028, 029, 030).
- The `transaction()` infrastructure came in via rebase from
  `feature/svelte-frontend` (commit `15f6292`). That commit is the
  precondition for this work; the proposal's "Add context managers to
  `DBConnectionFactory`" section was retired by the rebase.
