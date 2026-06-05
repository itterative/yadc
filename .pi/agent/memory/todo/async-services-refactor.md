---
name: async-services-refactor
description: API endpoints call heavy sync functions from async handlers, blocking the Quart event loop. Deferred refactor to make services async-native.
category: backend
priority: 2
keep_updated: true
---

# Async-Sync Bridge in API Services

## Problem

Quart `async def` endpoints in `yadc/api/controllers/` call heavy synchronous functions without `asyncio.to_thread()` or `loop.run_in_executor()`. There are **none** of those calls anywhere in the codebase. This blocks the event loop during:

- PIL image operations (thumbnail generation, `Image.open` in dataset scanning)
- Bulk file I/O (upload writes, `shutil.move`, `shutil.rmtree`)
- TOML parsing/serialization
- Jinja2 template rendering
- Keyring/crypto operations (`cmd_envs.load_env`)
- Full directory walks (`_scan_disk` opens every image with PIL)

Plain `def` routes are fine — Quart runs them in a thread pool automatically. The issue is strictly `async def` routes calling sync work.

## Worst Offenders (from controller audit)

| File | Endpoint | Blocking Work |
|---|---|---|
| `api_datasets.py` | `serve_image_thumbnail` | PIL `Image.open` + `thumbnail` + `save` |
| `api_datasets.py` | `add_dataset` | `import_dataset` / `create_dataset` → `_scan_disk()` |
| `api_datasets.py` | `upload_dataset` | `create_dataset_from_upload` → PIL validation + file writes + `_scan_disk()` |
| `api_datasets.py` | `commit_staged_upload` | `shutil.move` + `_scan_disk()` |
| `api_datasets.py` | `delete_dataset_items` | `rmtree` + `unlink` + `_scan_disk()` |
| `api_datasets.py` | `update_image_caption` | File reads/writes, `shutil.copy`, history rewrite |
| `api_datasets.py` | `preview_prompt` | Jinja2 render + TOML + glob |
| `api_export.py` | `export_dataset` | `resolve_dataset` (PIL every image) + `run_export` (read all captions) |
| `api_captioning.py` | `start_captioning` | `cmd_envs.load_env` + `preflight_images` → `load_dataset_config` → `resolve_dataset` |
| `api_configs.py` | `put_config` / `patch_config` | `rescan_dataset` → `_scan_disk()` |

## SQLite Safety

Wrapping sync calls in `asyncio.to_thread()` is **safe** for SQLite in this project because:
- `journal_mode=wal` allows concurrent readers + one writer
- `busy_timeout=5000` prevents immediate lock errors
- Each `with db.connection():` opens a new connection (Serialized mode, thread-safe)

**Trap**: `DBConnectionFactory` uses a `ContextVar` for transaction tracking. ContextVars don't propagate across thread boundaries. If a transaction starts in the async thread and inner work moves to `to_thread()`, the inner code sees `ContextVar = None` and opens a second connection. Always wrap the **entire** operation (including its `with db.transaction():` boundary) in `to_thread()`, never split a transaction across threads.

## Deferred Approach

Rather than scattered `to_thread()` wrappers in controllers, the cleaner long-term fix is:

1. **Keep sync methods** in `DatasetService`, `ManagedDatasetsService`, etc. for background threads and CLI reuse.
2. **Add thin `async def *_async()` wrappers** in services (e.g. `rescan_dataset_async()`) that call `asyncio.to_thread(self.rescan_dataset, ...)`.
3. **Controllers `await` the `*_async()` variants** instead of calling sync methods directly.
4. **Core/`cmd/` code stays sync** — wrap at the controller call site when no service wrapper exists (e.g. `resolve_dataset`, `run_export`).

This avoids duplicating `to_thread()` boilerplate across multiple controllers and keeps transactional units intact.

## Decision

**Deferred.** The current pattern (sync routes where possible, async routes with blocking calls) hasn't caused visible issues in practice. If the API becomes sluggish under concurrent load, this is the first thing to fix. At that point, start with `DatasetService` wrappers and the export endpoint.

## Key Files

- `yadc/api/controllers/api_datasets.py`
- `yadc/api/controllers/api_export.py`
- `yadc/api/controllers/api_captioning.py`
- `yadc/api/controllers/api_configs.py`
- `yadc/api/services/datasets.py`
- `yadc/api/services/managed_datasets.py`
- `yadc/api/services/dataset_upload.py`
- `yadc/api/modules/db_connection_factory.py`
