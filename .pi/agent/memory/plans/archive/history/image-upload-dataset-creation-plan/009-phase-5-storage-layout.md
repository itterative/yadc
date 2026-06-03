# Phase 5: Separate Root and Folder Storage

## Summary

Refactored `create_dataset_from_upload()` to store root files and folder-uploaded files in separate directories (`images/` and `folders/`), with one `[[dataset]]` config entry per top-level folder.

## Changes

### Modified Files

- `yadc/api/services/datasets.py`:
  - Replaced single `data/` directory with `images/` (flat) and `folders/` (structured).
  - Root files (no `/` in path, e.g. from "Choose Files") → written flat to `images/`.
  - Folder files (has `/` in path, e.g. from `webkitdirectory` or drag-and-drop folder) → written to `folders/` preserving relative structure.
  - Config TOML generation:
    - One `[[dataset]] path = "images"` entry if any root files were uploaded.
    - One `[[dataset]] path = "folders/<name>"` entry per unique top-level folder in `folders/`.
  - Updated docstring to describe the new behavior.

## Storage Layout

```
STATE_PATH/<dataset-name>/
  config.toml
  images/
    a.jpg
    b.png
    c.txt
  folders/
    train/
      x.jpg
      y.toml
    val/
      z.jpg
```

## Config TOML Example

```toml
[[dataset]]
path = "images"
[[dataset]]
path = "folders/train"
[[dataset]]
path = "folders/val"
```

## Verification

- `uv run ruff check yadc/api/services/datasets.py` → passed
- `uv run pytest tests` → 128 passed
