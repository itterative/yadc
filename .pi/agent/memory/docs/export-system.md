---
name: export-system
description: How the export system works — backends, formats, zip export, and the draft/caption source selection.
category: architecture
---

# Export System

## Overview

The `export` CLI command exports captioned datasets to formats compatible with training tools. The Web UI also supports exporting as a downloadable zip archive (with optional images).

## Backend System (`core/exporters/`)

- `list_backends()` — returns dict of backend name → `_BackendDescriptor`
- `get_backend(name)` — returns single `_BackendDescriptor`
- `run_export(name, images, **opts)` — dispatches to backend's `run()` function (writes to filesystem)
- `stream_export_zip(name, images, **opts)` — dispatches to backend's `iter_zip_members()` and returns `(Iterator[bytes], count)` for a streaming zip (no `Content-Length`; hypercorn emits `Transfer-Encoding: chunked`). Pre-flight (stat every file, read captions) runs eagerly inside it, so a missing source raises before any bytes are streamed.

Each backend is a module with a `Backend` dataclass (`name`, `description`, `formats`, `zip_only`), a `run()` function, and an `iter_zip_members()` function. Register a new backend by importing its module and adding it to `_BACKENDS` in `__init__.py`.

`zip_only=True` marks a backend whose only output is a zip (e.g. yadc). The API forces `format` to its single format and requires `zip=True`; the backend's `run()` raises. The `/export/backends` response surfaces `zip_only` so the frontend can hide irrelevant controls (source/drafts/extension) and force zip.

## sd-scripts Backend

Transforms captions into training-tool metadata.

**Formats**: `json`, `jsonl`, `txt`

| Format | Output | Description |
|--------|--------|-------------|
| `json` | `metadata.json` | JSON dict keyed by filename → `{"caption": "..."}` |
| `jsonl` | `metadata.jsonl` | One JSON object per line: `{"image_path": "...", "caption": "..."}` |
| `txt` | per-image caption files | Writes caption text files alongside or to output directory |

## yadc Backend

Raw archival: bundles each image together with **all** of its on-disk sidecar files — caption (`.txt`), drafts (`.<name>.draft~`), metadata (`.toml`), backup (`.toml~`), history (`.history~`) — preserving yadc's native layout. Round-trippable (faithful backup of the dataset).

**Format**: `zip` only. **`zip_only=True`.** Sidecars are discovered by globbing `<image-stem>.*` in the image's directory (catches every sidecar type, no hardcoded suffix list); the literal `.` after the stem avoids matching a longer filename like `img0010.png` for an `img001` image. Images are always included; `source`/`drafts`/`include_images` are ignored (it archives everything raw). The shared `relative_arc_name(path, base_dir)` helper (in `utils.py`, also used by sd-scripts) computes arc names.

## Zip Export (Web UI)

When the `zip` field is set in the export request body, the API returns the export as a streaming `application/zip` response instead of writing to the filesystem.

- **`iter_zip_members()`** in each backend runs pre-flight (captions + file stats) and returns a lazy iterator of `ZipMember` + a count; the shared `stream_zip_bytes()` (in `zip_stream.py`) encodes them via the `stream-zip` library.
- **Compression**: every member is **buffered zip64 STORED** (`NO_COMPRESSION_64`) — stream-zip computes each entry's CRC32 + size itself and buffers only that one member, so memory is bounded to the largest single member (an image ≈ ≤20 MB), not the dataset. zip64 keeps it correct for >4 GB archives. No DEFLATE: images are incompressible and STORED keeps `zip size ≈ Σ member sizes`, which the export dialog's size estimate relies on. The file-reading seam lives in `zip_stream.py` so a future async-I/O move is localized there.
- **`include_images`** — when true, source image files are bundled into the zip at their relative paths from the dataset base dir.
- **`base_dir`** — parent of `config.toml`, used to compute relative paths inside the zip (e.g. `images/cat.jpg`, `folders/train/img.jpg`).
- The `relative_arc_name()` helper (in `utils.py`, also used by yadc) computes the arc name, falling back to the bare filename for external datasets.
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
