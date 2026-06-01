---
date: 2026-06-01
---
# Structural Split Recommendations

A review of the plan's implementation found several files that have grown well past comfortable size. Recommendations #1 and #2 (DatasetConfig.svelte and dataset_upload.py) were implemented in this commit. The remaining recommendations are deferred for future cleanup passes.

## File sizes as of this entry

| File | Lines | Notes |
|------|-------|-------|
| `yadc/api/services/datasets.py` | 1320 | Pre-existing, grew ~250 lines from this plan |
| `yadc/webui/src/lib/components/datasets/DatasetConfig.svelte` | 796 | **New** (Phase 8.1) — biggest new file |
| `yadc/webui/src/routes/datasets/[name]/CaptionSettings.svelte` | 652 | Pre-existing, reduced by Phase 4 |
| `yadc/api/services/dataset_upload.py` | 693 | **New** (Phase 8.2) — biggest new backend file |
| `yadc/webui/src/lib/components/dataset/ImageDetail.svelte` | 575 | Pre-existing, grew slightly |
| `yadc/api/controllers/api_datasets.py` | 416 | Pre-existing, grew ~150 lines |
| `yadc/webui/src/lib/components/datasets/DatasetUploadPanel.svelte` | 383 | **New** (Phase 8.4) |
| `yadc/api/controllers/api_configs.py` | 336 | Pre-existing, grew (history endpoints) |
| `yadc/webui/src/lib/components/settings/CaptionOptionsFields.svelte` | 335 | **New** (Phase 4) |
| `yadc/webui/src/lib/stores/configs.ts` | 257 | Pre-existing, grew (config history) |
| `yadc/webui/src/lib/components/datasets/ConfigHistory.svelte` | 205 | **New** (Phase 5) |
| `yadc/webui/src/lib/components/ui/KeyValueEditor.svelte` | 193 | **New** (Phase 1) |

## Recommendation 1: Split `DatasetConfig.svelte` (796 lines) ✅ IMPLEMENTED

The file owns 16 field states, a parse/serialize round-trip via two HTTP methods, three tab views, a debounced live preview, and dual-view dirty tracking. The script block alone is ~330 lines.

**Implemented split:**

```
yadc/webui/src/lib/components/datasets/
  DatasetConfig.svelte          # Container — CompactPillTabs + footer (save/reload) only
  DatasetConfigForm.svelte      # The structured form (was the bulk of the template)
  DatasetConfigAdvanced.svelte  # Raw TOML editor + dirty banner
  ConfigHistory.svelte          # (already separate — kept)
  datasetConfig/
    state.svelte.ts             # $state module: fields, loaded snapshots, dirty derivations
    patch.ts                    # buildPatch() + preview scheduling (pure functions)
```

**Trade-off acknowledged:** Svelte 5 state modules with `$state` are still new in the ecosystem. The migration was sanity-checked by extracting one property at a time and verifying reactivity across `DatasetConfigForm`.

## Recommendation 2: Split `dataset_upload.py` (693 lines) ✅ IMPLEMENTED

Three concerns in one service:
1. **Validation helpers** (`_validate_image`, `_validate_toml`, `_unique_path`, `_cleanup_staging_dirs`) — pure, no state
2. **Create-from-upload** — validates, writes, registers
3. **Staging/commit** — the conflict-detection + commit-resolve flow

Plus the validation/orphan-sidecar logic was **duplicated** between `create_dataset_from_upload` and `append_dataset_from_upload`.

**Implemented split:**

```
yadc/api/services/
  dataset_upload.py             # Public service — orchestrates create/append/commit
  dataset_upload_validation.py  # Pure validators + path utilities (no class state)
  dataset_upload_staging.py     # Staging dir helpers, conflict detection, sidecar grouping
```

The duplicated orphan-sidecar check was deduplicated by extracting a shared helper used by both flows.

## Recommendation 3: Extract managed-dataset operations from `datasets.py` (1320 lines) ⏳ DEFERRED

The Phase 8 additions (`delete_items` at ~180 lines, `list_folders` at ~50 lines, plus path resolution helpers) are all about managed datasets and don't fit the file's "TOML-based dataset scanning" core.

**Proposed split:**

```
yadc/api/services/
  datasets.py                   # Core: list_datasets, scan, register, image listing
  managed_datasets.py           # delete_items, list_folders, path resolution, MANAGED_* constants
```

**Why deferred:** This is more disruptive than #1 or #2 because `DatasetService` is referenced from many call sites. Worth doing but in a dedicated PR.

## Recommendation 4: Trim `DatasetUploadPanel.svelte` (383 lines) ⏳ DEFERRED

The file mixes file selection UI, upload progress, and the conflict-resolution sub-flow.

**Proposed split:**

```
yadc/webui/src/lib/components/datasets/
  DatasetUploadPanel.svelte     # Container — orchestrates phases
  ConflictResolution.svelte     # The "conflicts" phase: list, per-file action, bulk set, commit
```

**Why deferred:** Container is at the edge of comfort. Worth doing only if conflict-resolution logic grows.

## Recommendation 5: Split `stores/configs.ts` (257 lines) ⏳ DEFERRED

Currently mixes: config CRUD, config history, export backends, and dataset drafts.

**Proposed split:**

```
yadc/webui/src/lib/stores/
  configs.ts          # Config CRUD only
  configHistory.ts    # fetchConfigHistory, restoreConfigHistory
  exports.ts          # fetchExportBackends, runExport
  datasetDrafts.ts    # fetchDatasetDrafts
```

**Why deferred:** Low priority. Each sub-store is small enough that the bundling isn't a real cost.

## Recommendation 6: Other tweaks ⏳ DEFERRED

- **`api_configs.py` (336 lines)** — history endpoints could move to `api_config_history.py` for symmetry, but the file is small enough.
- **`CaptionOptionsFields.svelte` (335 lines)** — at the edge; only split if it grows.
- **`api_datasets.py` (416 lines)** — folder management and item-deletion routes could move to `api_managed_datasets.py` to align with Recommendation 3.

## Cross-cutting concern: validation duplication

`dataset_upload.py` had the same orphan-sidecar check duplicated in `create_dataset_from_upload` and `append_dataset_from_upload`. Recommendation 2's split fixed this naturally.

`KeyValueEditor.svelte` value-type conversion logic (lines 41-67) is also a self-contained module that could move to a `keyValueConversions.ts` helper if it ever needs to be tested in isolation.

## Prioritization summary

| # | Recommendation | Status |
|---|----------------|--------|
| 1 | Split `DatasetConfig.svelte` | ✅ Implemented |
| 2 | Split `dataset_upload.py` | ✅ Implemented |
| 3 | Extract from `datasets.py` | ⏳ Deferred |
| 4 | Trim `DatasetUploadPanel.svelte` | ⏳ Deferred |
| 5 | Split `stores/configs.ts` | ⏳ Deferred |
| 6 | Other tweaks | ⏳ Deferred |
