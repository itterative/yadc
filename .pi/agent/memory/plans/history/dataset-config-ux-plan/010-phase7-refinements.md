---
date: 2025-05-30
---
# Phase 7: Refinements (partial)

**Context:** Post-implementation polish for issues discovered during use.

### 7.1 Extras editing dirty tracking ✅

**Problem:** Changing extras values in `KeyValueEditor` didn't enable the "Save Config" button. The preview DID update though.

**Root cause:** `populateFields()` in `DatasetConfig.svelte` assigned `datasetEntries = loadedDatasetEntries = sameArray`. Both `$state` variables shared the same underlying objects. When `bind:entries` mutated `datasetEntries[i].extras`, the change was visible through `loadedDatasetEntries[i].extras` too — so `entriesEqual()` always returned `true`.

**Fix:** One-line fix in `populateFields()` — deep-copy entries for the loaded snapshot:
```js
datasetEntries = entries;
loadedDatasetEntries = entries.map((e) => ({
    ...e,
    extras: e.extras.map((kv) => ({ ...kv }))
}));
```

Also removed stale `onconfigsaved` prop from `+page.svelte` (SidePanel no longer has it after user moved history tab to DatasetConfig).

### 7.2 Snapshot strategy ✅

**Problem:** History only saved pre-write (old) content. Initial config state was never recorded. Users couldn't revert to the current version.

**Fix:**
- **PUT/PATCH**: Moved `save_snapshot` to after the write, saving the new content (result).
- **Import/Create/Upload**: Added `_snapshot_initial()` helper in `api_datasets.py` — reads config file and saves snapshot after registration. Called on import, create, and after upload "complete" event.
- **Restore**: Keeps pre-restore snapshot (captures state being replaced, enables undo). No post-restore snapshot (content was already in history; next save will capture it).
- **tomlkit Pydantic fix**: tomlkit wraps values in `Bool`/`Integer`/`String` types that Pydantic's Rust validation rejects. Added `toml_to_plain()` in `dict_utils.py` — recursively converts tomlkit containers to plain Python types before validation. Used in GET validation, PATCH validation, and response `parsed` fields.
- **Unit tests**: 24 new tests in `tests/utils/test_dict_utils.py` covering `toml_merge`, `toml_to_plain`, comment preservation, and Pydantic compatibility.

**Files:**
- `yadc/api/controllers/api_configs.py` — post-write snapshots, `toml_to_plain` for validation
- `yadc/api/controllers/api_datasets.py` — `ConfigHistoryService` injection, `_snapshot_initial()`
- `yadc/utils/dict_utils.py` — `toml_to_plain` and `_list_to_plain`
- `yadc/webui/src/routes/datasets/[name]/DatasetConfig.svelte` — `populateFields` deep-copy fix
- `yadc/webui/src/routes/datasets/[name]/+page.svelte` — removed stale `onconfigsaved` prop
- NEW: `tests/utils/test_dict_utils.py` — 24 unit tests
