---
date: 2026-05-31
---
# Delete Items Endpoint for Managed Datasets

## Implementation

### Backend: `DatasetService.delete_items()`

Added `delete_items(name: str, paths: list[str]) -> tuple[list[str], list[str]]` to `DatasetService`:

- **Guard check**: Dataset must exist and have `source == "upload"`. Raises `ValueError` otherwise.
- **Path resolution**: For each relative path, tries:
  1. File in `images/<path>`
  2. File in `folders/<path>`
  3. Directory in `folders/<path>`
- **Sidecar cleanup**: `_delete_file_with_sidecars()` removes `.txt`, `.toml`, `.history~`, and `.draft~` files matching the image stem
- **Batch support**: Multiple paths in one call; returns `(deleted, warnings)`
- **Auto-rescan**: Calls `rescan_dataset()` after deletions to update SQLite index

### Backend: `DELETE /datasets/<name>/items`

New endpoint in `api_datasets.py`:

```json
// Request
DELETE /api/datasets/my-dataset/items
{ "paths": ["foo.jpg", "train/img.jpg", "train"] }

// Response
{ "deleted": ["foo.jpg", "train"], "warnings": ["train/img.jpg: Not found"] }
```

### Frontend: `ImageDetail.svelte`

- Added **Delete** button next to Caption/Edit (only for `source === 'upload'`)
- Confirmation dialog: `"Delete "filename" and its sidecars? This cannot be undone."`
- Computes relative delete path from absolute `item.path` using regex (`/images/` or `/folders/`)
- Calls `ondelete()` callback after successful deletion

### Frontend: `SidePanel.svelte` + `+page.svelte`

- Added `onimagedelete` prop through the component tree
- On delete: clears `focusedItem`, switches panel to Caption tab, reloads image grid

### Tests

5 new tests in `test_dataset_upload_service.py`:
- `test_delete_image_and_sidecars` — image + `.txt` + `.toml` all removed
- `test_delete_folder` — entire directory recursive delete
- `test_delete_missing_path_warns` — graceful handling of missing paths
- `test_delete_non_managed_dataset_rejects` — guard check rejects non-upload datasets
- `test_delete_mixed_batch` — files + folders + missing in one batch

All 211 tests pass.
