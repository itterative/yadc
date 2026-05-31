---
date: 2026-05-31
---
# Folder Management Tab for Managed Datasets

## Problem

Users need a way to delete entire folders from managed datasets without editing config paths (which is now blocked). Individual image deletion works, but deleting 50 images in a folder one-by-one is tedious.

## Solution

A dedicated **Manage** tab in `EditDatasetDialog.svelte`, only shown for managed datasets (`source === 'upload'`). The Manage tab contains sections, starting with a **Folders** section.

### Backend

- `GET /datasets/<name>/folders` — returns folder list with image counts and `can_delete` flag
- `DatasetService.list_folders(name)` — scans `images/` and `folders/*/` on disk
- `DatasetService.delete_items(name, paths)` — accepts **prefixed paths**:
- `DatasetService._remove_folder_dataset_entries()` — removes `[[dataset]]` entries for deleted folders from config TOML before rescanning
  - `images/foo.jpg` → root file
  - `folders/train/foo.jpg` → folder file
  - `folders/train` → folder directory
  - `images` → rejected (root images dir is protected)
  - Bare paths still work as backward-compat fallback
- `ImageInfo.delete_path` — computed by backend from absolute path + config location. Frontend uses this directly instead of parsing paths.

This removes ambiguity: a folder literally named `images` inside `folders/` is addressed as `folders/images`, distinct from the root `images/` directory.

### Constants

Module-level constants in `datasets.py`:
- `MANAGED_IMAGES_PREFIX = "images"`
- `MANAGED_FOLDERS_PREFIX = "folders"`

Used by `list_folders`, `delete_items`, `_compute_delete_path`, and tests.

### Frontend

- New `DatasetManageTab.svelte` component in `lib/components/datasets/`
- Lists folders as cards: name, image count, delete button (controlled by `can_delete`)
- Root `images` shown without delete button
- Delete uses `confirmDialog.danger()` → `deleteDatasetItems()` → refresh
- `ImageDetail.svelte` uses `item.delete_path` directly — no frontend path parsing
- Manage tab added to `EditDatasetDialog.svelte` alongside Config and Upload

### Note on previous approach

Initially implemented as a "Folders" tab inside `DatasetConfig.svelte`, but moved to a separate "Manage" tab at the dialog level to keep config editing separate from storage management.
