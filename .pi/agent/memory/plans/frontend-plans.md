---
name: frontend-plans
description: Frontend implementation status, component index, and design notes for the webui.
---

# Frontend Plans

All phases (1–4) are **implemented**.

For frontend directory structure, stores, component organization, and general patterns — see `frontend-architecture` memory.

## Backend Controllers

| File | Endpoints |
|------|-----------|
| `api_datasets.py` | Dataset/image CRUD, caption update, prompt preview, **add dataset (import TOML / create new)** |
| `api_captioning.py` | Start/stop/status (SSE) captioning |
| `api_envs.py` | Env CRUD + model list proxy + key-mode get/set + per-field value reveal |
| `api_templates.py` | Template CRUD + variable extraction |
| `api_configs.py` | Dataset config TOML CRUD (GET, PUT raw TOML, **PATCH JSON merge**, DELETE) |
| `api_export.py` | Export backends list + run export |
| `api_events.py` | Global SSE stream |

## Component Design Notes

- **CodeMirror.svelte** (`ui/`): Three separate `$effect` blocks (create/destroy/sync) — never combine. See notes below in this section. Uses `editable` prop (default `true`) — not `readonly`. Includes a `baseTheme` (dark surface, accent-colored selection via `color-mix(in oklch, ...)`) and a `darkHighlightStyle` (`HighlightStyle.define` + `syntaxHighlighting`) that maps all `@lezer/highlight` tags to CSS `--color-syn-*` variables defined in the Tailwind `@theme` block.
- **TomlEditor.svelte** (`ui/`): Wraps CodeMirror with TOML syntax, line wrapping, hidden gutters. `editable` prop (default `true`). Used in `ImageDetail` for TOML extras editing, `PromptPreview` for readonly template context display, and `EditDatasetDialog` for dataset config editing.
- **JinjaEditor.svelte** (`ui/`): Wraps CodeMirror with Jinja2 syntax, variable extraction bar. Also uses `editable` prop (default `true`).
- **PromptPreview.svelte** (`ui/`): Self-contained prompt preview — manages its own template loading, preview fetching, and expand/collapse state. Used in `ImageDetail`.
- **EnvSelector.svelte** (`settings/`): Environment form with env dropdown, URL/token/model fields, model fetching. Uses `$bindable()` props for two-way value binding with parent. **Auto-fetches models** when env loads (populates dropdown immediately). The "Manage…" link opens `SettingsDialog` at the Environments tab via the `settingsDialog` store.
- **ConfigEditor.svelte** (`settings/`): *(REMOVED)* Was a full config CRUD panel. Dataset config management is now on the dataset listing page (edit/delete buttons), the side panel Config tab (`DatasetConfig.svelte`), and caption settings panel (config defaults integration).
- **TemplateManager.svelte** (`settings/`): Full template CRUD panel — sidebar list + JinjaEditor + new/save/delete. Self-contained, takes `open` prop to trigger data loading.
- **ToastItem.svelte** (`ui/`): Variant icons (info/success/warning/error) rendered via dedicated SVG icon components (`SvgInfo`, `SvgCheck`, `SvgWarning`, `SvgError`) instead of text characters.

## Known Issues

- No per-tile SSE updates during captioning (aggregate events only)
- **Topbar padding / height inconsistency:** The dataset listing (`#/`) and templates (`#/templates`) pages feel cramped below the topbar because `.app-topbar` has `padding-bottom: 0`. Adding bottom padding globally causes the dataset browser (`#/datasets/:name`) to shift down because its subtitle row uses `min-h-7` to reserve space for the captioning status row, which is taller than the idle text line. The hamburger button and subtitle also shift slightly when captioning starts/stops. **Superseded by `captioning-status-bar-plan`** — moving the progress UI out of the topbar entirely.
