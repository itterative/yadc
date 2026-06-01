---
date: 2026-06-01
---
# Managed Datasets Cycle Fix

## Summary

Follow-up to the `datasets.py` split (history entry 028, Recommendation 3). The
initial implementation introduced a two-way constructor dependency between
`DatasetService` and `ManagedDatasetsService`:

- `DatasetService.__init__` took `managed: ManagedDatasetsService` so it could
  call `self._managed.compute_delete_path(...)` when building `ImageInfo`.
- `ManagedDatasetsService.__init__` already took `datasets: DatasetService` for
  `get_dataset` and `rescan_dataset`.

The cycle was hidden by `TYPE_CHECKING` (no runtime import) and by lazy
imports, but it was still evident from the constructor signatures and made
basedpyright complain about a cycle it couldn't fully resolve.

## The fix

`compute_delete_path` is a pure function with no service dependencies. It
belongs in a layout-utility module, not on any service. The cycle breaks
cleanly once it's moved out.

### New module: `yadc/api/services/managed_paths.py` (68 lines)

Owns the on-disk layout convention for managed datasets:

- Layout constants: ``MANAGED_IMAGES_PREFIX``, ``MANAGED_FOLDERS_PREFIX``
- Path builders: ``managed_base_dir``, ``managed_images_dir``, ``managed_folders_dir``
- The path resolver: ``compute_delete_path(image_path, config_path)``

This is the canonical home for the convention. Services consume it; nothing
else re-exports these symbols.

### Updated module: `yadc/api/services/managed_datasets.py` (274 lines)

- Dropped the ``MANAGED_*`` constants and the ``compute_delete_path`` method.
- Imports both from ``managed_paths`` instead.
- Uses ``managed_images_dir(config_path)`` / ``managed_folders_dir(config_path)``
  in ``delete_items`` and ``list_folders`` to replace the inlined
  ``base_dir / MANAGED_IMAGES_PREFIX`` constructions.
- Still takes ``datasets: DatasetService`` in the constructor (one-way
  dependency on the foundational service).

### Updated module: `yadc/api/services/datasets.py` (1086 lines)

- Dropped the ``managed: ManagedDatasetsService`` constructor parameter.
- The two ``self._managed.compute_delete_path(...)`` call sites in
  ``list_images`` and ``get_image`` now call the free function
  ``compute_delete_path(...)`` imported from ``managed_paths``.
- Dropped the re-export of ``MANAGED_*`` constants. Tests now import them
  from ``managed_paths`` directly.

### Updated module: `yadc/api/services/dataset_upload_staging.py`

- Replaced string literals ``"images"`` and ``"folders"`` with the
  ``MANAGED_IMAGES_PREFIX`` / ``MANAGED_FOLDERS_PREFIX`` constants from
  ``managed_paths`` for consistency.

## Dependency graph after the fix

```
ManagedDatasetsService
  └─→ DatasetService           (for get_dataset, rescan_dataset)
  └─→ DatasetWatcherService    (for expect_file_change)
  └─→ managed_paths            (for constants and path builders)

DatasetService
  └─→ DatasetWatcherService
  └─→ DBConnectionFactory
  └─→ Configuration
  └─→ EventDispatcher
  └─→ managed_paths            (for compute_delete_path)

managed_paths  (no service deps, pure utilities)
```

The two-way cycle becomes a one-way dependency: ``DatasetService`` is
foundational, ``ManagedDatasetsService`` builds on top. Normal layering.

## Architectural principle articulated in the review

> Services provide the functionality; the others help the services.

Applied here: ``managed_paths`` (helper) is the canonical source for the
layout convention. ``managed_datasets`` (service) and ``datasets`` (service)
both consume it as consumers. Nothing re-exports the constants or helpers
through the services — that would re-introduce the coupling we just broke.

## Verification

- All 253 tests pass (no test changes beyond updating the constant import
  path in ``test_dataset_upload_service.py``).
- ``ruff check`` clean on the new and modified files.
- ``ruff format --check`` clean.
- basedpyright: 0 errors, 16 pre-existing warnings (no new warnings
  introduced).

## Open follow-up

The string literals ``"images"`` and ``"folders"`` in ``dataset_upload.py``
itself (lines 126, 127, 225, 270, 271, 274, 275, 418, 419, 493) could also
be replaced with the constants for full consistency. The user noted this
was "less of a concern" and it wasn't done in this pass — flagged as a
polish item.

## File sizes after the refactor

| File | Lines | Role |
|------|-------|------|
| `managed_paths.py` | 68 | Layout convention (constants + path builders + resolver) |
| `managed_datasets.py` | 274 | Managed-dataset service (delete_items, list_folders, sidecar cleanup) |
| `datasets.py` | 1086 | Core dataset service (scanning, indexing, image queries) |

`datasets.py` was 1320 lines before the recommendation, 1210 after the
initial split, and 1086 after the cycle fix — three operations
combined removed ~234 lines.
