---
name: export-system
description: How the export system works — backends, formats, and the draft/caption source selection.
---

# Export System

## Overview

The `export` CLI command exports captioned datasets to formats compatible with training tools.

## Backend System (`core/exporters/`)

- `list_backends()` — returns dict of backend name → Backend dataclass
- `get_backend(name)` — returns single Backend
- `run_export(name, images, **opts)` — dispatches to backend's `run()` function

Each backend is a module with a `Backend` dataclass and a `run()` function.

## sd-scripts Backend (only backend currently)

**Formats**: `json`, `jsonl`, `txt`

| Format | Output | Description |
|--------|--------|-------------|
| `json` | `metadata.json` | JSON dict keyed by filename → `{"caption": "..."}` |
| `jsonl` | `metadata.jsonl` | One JSON object per line: `{"image_path": "...", "caption": "..."}` |
| `txt` | per-image caption files | Writes caption text files alongside or to output directory |

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
