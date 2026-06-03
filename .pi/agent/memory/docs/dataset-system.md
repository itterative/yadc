---
name: dataset-system
description: Dataset subsystem end-to-end — Web UI dataset model, three creation flows, managed dataset layout, upload pipeline, diff-scan rescan, background refresh, DatasetImage persistence. Cross-cuts services, controllers, and filesystem conventions.
category: architecture
keep_updated: true
---

# Dataset System

The end-to-end dataset subsystem — from dataset creation through image persistence. For inotify event tracking and frontend suppression, see `dataset-watcher`. For platformdirs paths and DatasetImage file conventions, see `paths-and-storage`. For DI/repository layering, see `api-di-system` and `repository-pattern`.

## Web UI Dataset Model

A "dataset" IS a TOML config file. There is no `datasets` table that stores paths — the file is the source of truth.

- **Created / uploaded datasets** live at `STATE_PATH/datasets/<name>/config.toml`.
- **Imported datasets** reference an external TOML path (the original file is left in place, relative paths resolved to the state dir copy).
- `name` is the unique key — no separate `path` column.

The state-dir layout for a created dataset `<name>`:

```
STATE_PATH/datasets/<name>/
  config.toml         # the dataset config (TOML)
  images/             # (uploaded only) root files, flat
  folders/<dir>/      # (uploaded only) folder uploads, one subdir per top-level folder
  .staging/<uuid>/    # (uploaded only) staging area for in-progress appends
```

## Three Creation Flows

All three live in `yadc/api/services/datasets.py` and are exposed by the `DatasetService`.

1. **`import_dataset(name, toml_path)`** — copies an existing TOML file to the state dir, resolving relative paths to point at the new location. Used when the user has a config on disk and wants to manage it through the webui.

2. **`create_dataset(name, image_paths)`** — generates a new TOML at `STATE_PATH/datasets/<name>/config.toml` pointing to external directories the user provided. The dataset is `source="create"` — images stay where they are, only the config is in the state dir.

3. **`create_dataset_from_upload(name, files)`** — creates a self-contained dataset from uploaded files (images + sidecars). Stores files under `STATE_PATH/datasets/<name>/{images,folders}/...`. The dataset is `source="upload"` (managed). Returns `DatasetUploadResult(dataset, warnings)`.

## Managed Dataset Layout

For uploaded (`source="upload"`) datasets, the layout is `<config_dir>/{images,folders}/...` where `<config_dir>` is `STATE_PATH/datasets/<name>/`.

- Root-level files go to `images/` (flat).
- Top-level folders go to `folders/<dir>/` (one subdir each, recursively containing the folder's contents).

**Constants and path builders** live in `yadc/api/services/managed_paths.py`:

- `MANAGED_IMAGES_PREFIX = "images"`
- `MANAGED_FOLDERS_PREFIX = "folders"`
- `managed_base_dir(name)`, `managed_images_dir(name)`, `managed_folders_dir(name)` — directory builders
- `compute_delete_path(name, relative_path)` — populates `ImageInfo.delete_path` for the image-listing endpoint

This module is the single source of truth for the layout — no service deps, safe to import from anywhere. Pure layout math, not a service.

## Upload Pipeline

Three endpoints, all in `yadc/api/controllers/api_datasets.py`. All accept an optional `?source=<clientId>` query parameter to suppress self-triggered `DatasetChangedEvent` in the originating tab (threaded through `DatasetUploadService` calls).

### `POST /api/datasets/upload` (create)

`multipart/form-data` with `name` + `files`. Two-stage image validation:

1. `Image.verify()` — fast header check
2. `Image.load()` — fallback for headers that pass verify but fail on full decode

Plus TOML syntax validation, orphan sidecar detection (`.<name>.draft~` for a file that wasn't uploaded), and atomic cleanup on failure (no partial state). Configurable `max_upload_size_bytes` limit from `Configuration`.

### `POST /datasets/<name>/upload` (append)

Writes uploaded files to a temporary `.staging/<uuid>/` directory under the dataset. Staging dirs older than `STAGING_MAX_AGE_SECONDS = 24 * 3600` are cleaned up by a periodic `JobScheduler` job in `DatasetUploadService.__init__`.

### `POST /datasets/<name>/staging/commit` (resolve conflicts)

Moves staged files into the live managed layout, applying user-chosen conflict resolution (skip / overwrite / rename). Each `shutil.move` is preceded by a `DatasetWatcherService.expect_file_change(name, str(dest_path), source=source)` call so the resulting `DatasetChangedEvent` is suppressed in the originating tab.

## Upload Source-ID Propagation

The `?source=<clientId>` query parameter is threaded through all three endpoints as a kwarg to `DatasetUploadService`. For every file write (create) and `shutil.move` (commit), the service calls:

```python
watcher.expect_file_change(name, str(dest_path), source=source)
```

The final `rescan_dataset(...)` is also called with `source=source`. The result is that the originating tab's `DatasetChangedEvent` carries its own `clientId`, and the SSE listener in `events.ts` suppresses it. Without this, a drop-and-upload would show a spurious "files have changed" refresh banner to the user who just initiated the upload. See `dataset-watcher` for the suppression pattern.

## Dataset Rescan (Diff Scan)

`DatasetService._apply_disk_scan` does a **diff scan** — compares the freshly-walked disk state against the current index and only writes SQL for changed rows (`to_upsert` + `to_delete`). Returns `bool` (whether anything changed).

Callers:

- `_refresh_stale_datasets` — background job (see below)
- `rescan_dataset(source=...)` — public API for manual rescan, dispatches `DatasetChangedEvent(job_id=source)` only when the scan found real changes

## Background Dataset Refresh

`DatasetService` runs a periodic `JobScheduler` job that calls `_apply_disk_scan` on all datasets, with interval from `Configuration.dataset_refresh_interval_seconds` (default 300s). Serialized with manual rescans via `DatasetService._refresh_lock` to prevent SQLite write-lock contention.

The `JobScheduler` is optional so tests can construct the service without it.

**Why both watcher and refresh?** The inotify watcher drives live updates in the common case. The stale-refresh job is a fallback for changes the watcher can't see — e.g. external edits to a dataset's `config.toml` that add new image paths, where no inotify event fires.

## DatasetImage Persistence

File conventions (extensions configurable per dataset, defaults shown) — full table in `paths-and-storage`:

| File | Default ext | Purpose |
|------|-------------|---------|
| `<stem>.txt` | `.txt` | Caption text |
| `<stem>.toml` | `.toml` | Metadata extras |
| `<stem>.toml~` | `.toml~` | Backup of previous TOML before update |
| `<stem>.history~` | `.history~` | Versioned TOML snapshots (entries separated by `----------` marker) |
| `<stem>.<name>.draft~` | `.<name>.draft~` | Named draft caption |

History is saved on every caption update (both captioning jobs and manual webui edits) and on extras updates. Drafts are saved on webui edits.

**Endpoints** (in `yadc/api/controllers/api_datasets.py`):

- `GET /images/<id>/history` — list history entries
- `PUT /images/<id>/history/<index>/restore` — restore an entry (saves current state to history first)
- `DELETE /images/<id>/history/<hash>` — delete a history entry by content hash (safe identification across renames)
- `DELETE /images/<id>/drafts/<name>` — delete a named draft
