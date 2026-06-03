---
name: yadc-config-v2
description: v2 dataset config format — [[dataset]] array-of-tables structure and v1 auto-conversion.
category: architecture
keep_updated: true
---

# yadc Config v2

## v2 Dataset Format
- `[[dataset]]` — array of tables (each is a `ConfigDatasetEntry`)
  - `path: str` — single directory path (not a list)
  - `images: list[DatasetImage]` — inline image overrides
  - `extras: dict[str, object]` — dataset-level template variables (defaults, per-image extras override)
- No `config_version` field — duck-typing: `isinstance(dataset_raw, list)` → v2, `isinstance(dataset_raw, dict)` → v1

## v1 → v2 Conversion
- `ConfigV1` / `ConfigDataset` for deserializing v1 TOML
- `ConfigV1.to_v2()` converts to `Config` with `ConfigDatasetEntry` list
- `parse_config(raw: dict) -> Config` is the entry point

## Dataset Resolution
- `yadc/core/dataset_resolver.py` — extracted from CLI, public API
  - `resolve_dataset(entries, caption_suffix, read_image=)` — scans paths, merges inline images, applies extras
  - `apply_extras_defaults(image, extras)` — applies dataset extras as defaults (per-image wins)
  - `reapply_dataset_extras(image)` — re-applies stored `_dataset_extras` after reconstruction (edit path)
  - `read_image_from_disk(path, caption_suffix)` — reads image + TOML from disk
- `DatasetImage._dataset_extras` — `PrivateAttr(default_factory=dict)`, stores the dataset-level extras for re-application
- CLI (`cli_caption.py`) calls `resolve_dataset()` and `reapply_dataset_extras()` in the edit path

## Key Files
- `yadc/core/config.py` — Config, ConfigDatasetEntry, ConfigV1, parse_config
- `yadc/core/dataset_resolver.py` — resolve_dataset, apply_extras_defaults, reapply_dataset_extras
- `yadc/core/dataset.py` — DatasetImage (has `_dataset_extras` PrivateAttr)
- `yadc/cli_caption.py` — `_load_dataset()` uses `parse_config()` + `resolve_dataset()`
- `tests/core/test_dataset_resolver.py` — 17 tests for resolution and extras merging
- `tests/core/test_config.py` — 10 tests for v1/v2 parsing

## Strict vs Relaxed Validation

`parse_config(raw)` is **strict by default** — it enforces CLI-level checks (`api.url` and `api.model_name` must exist, `prompt` must be specified). The webui provides these fields at caption time rather than at config edit time, so it calls `parse_config(raw, strict=False)` to skip those checks.

The flag is threaded via **Pydantic validation context**, not via fields on the model — so the same `Config` model behaves differently depending on caller. The `ConfigV1.to_v2()` migration uses `model_construct()` to avoid re-running validators on already-validated data.
