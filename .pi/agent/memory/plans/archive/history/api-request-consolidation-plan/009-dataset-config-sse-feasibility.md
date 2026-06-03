---
date: 2026-05-28
---
# Dataset Config SSE Sync: Feasibility Analysis

**Context:** After implementing SSE sync for environments (`config.toml`) and templates (`*.jinja`), we investigated whether the same pattern should apply to dataset configs.

**Findings:**

Dataset configs differ fundamentally from envs/templates in ways that make filesystem watching impractical:

1. **Arbitrary import paths** — `AddDatasetDialog.svelte` has an "Import TOML" mode that lets users point `importDataset(name, tomlPath)` to **any filesystem path**. The config is not required to live under `STATE_PATH`.

2. **Per-dataset state directories** — For configs created via "Create New", the config lives at `STATE_PATH/{name}/config.toml`. We'd need a watcher per dataset directory, not a single directory.

3. **No reactive store pattern** — `stores/configs.ts` has no `writable` store for config data. Components (`DatasetConfig.svelte`, `CaptionSettings.svelte`, `EditDatasetDialog.svelte`) fetch directly into local `$state`. Adding SSE would require either a reactive store refactor or per-component event handling.

4. **Low value** — The primary pain point (duplicate `fetchConfig` calls from sibling components) was already solved by debounce. Cross-tab dataset config sync is a rare use case.

**Conclusion:** Skip SSE for dataset configs. The combination of arbitrary paths, per-dataset directories, and lack of a reactive store makes the cost/benefit poor compared to environments and templates, which live in single well-known directories and already have reactive stores.
