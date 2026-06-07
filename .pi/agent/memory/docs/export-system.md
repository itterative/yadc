---
name: export-system
description: How the export system works — backends, formats, zip export, and the draft/caption source selection.
category: architecture
---

# Export System

## Overview

The `export` CLI command exports captioned datasets to formats compatible with training tools. The Web UI also supports exporting as a downloadable zip archive (with optional images).

## Backend System (`core/exporters/`)

- `list_backends()` — returns dict of backend name → Backend dataclass
- `get_backend(name)` — returns single Backend
- `run_export(name, images, **opts)` — dispatches to backend's `run()` function (writes to filesystem)
- `run_export_zip(name, images, **opts)` — dispatches to backend's `run_zip()` function (returns BytesIO zip buffer)

Each backend is a module with a `Backend` dataclass, a `run()` function, and a `run_zip()` function.

## sd-scripts Backend (only backend currently)

**Formats**: `json`, `jsonl`, `txt`

| Format | Output | Description |
|--------|--------|-------------|
| `json` | `metadata.json` | JSON dict keyed by filename → `{"caption": "..."}` |
| `jsonl` | `metadata.jsonl` | One JSON object per line: `{"image_path": "...", "caption": "..."}` |
| `txt` | per-image caption files | Writes caption text files alongside or to output directory |

## Zip Export (Web UI)

When the `zip` field is set in the export request body, the API returns the export as a streaming `application/zip` response instead of writing to the filesystem.

- **`run_zip()`** in each backend writes metadata into an in-memory zip via `zipfile.ZipFile`.
- **`include_images`** — when true, source image files are bundled into the zip at their relative paths from the dataset base dir.
- **`base_dir`** — parent of `config.toml`, used to compute relative paths inside the zip (e.g. `images/cat.jpg`, `folders/train/img.jpg`).
- The `_image_zip_path()` helper computes the relative arc name, falling back to the bare filename for external datasets.
- The zip variant uses `_collect_captions()` to avoid duplicating caption-reading logic.

For the `jsonl` and `json` zip variants, `image_path` values use the relative path (not just the filename) so that the zip is self-contained and training-ready.

## Source Selection

- `--draft <name>` — export a specific named draft instead of caption file
- `--with-draft <name>` — append additional drafts after primary source (can be repeated)
- Default: reads from caption files

## Format Inference

- If `--output` has `.json` → json, `.jsonl` → jsonl
- Otherwise defaults to `jsonl`
- `--format` overrides inference

## CLI Options

```
yadc export <dataset> [--backend sd-scripts] [--draft NAME] [--with-draft NAME]
                     [--format json|jsonl|txt] [--output PATH] [--append]
                     [--caption-extension .txt] [--env ENV] [--user-config NAME]
```
