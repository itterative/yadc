---
name: frontend-plans
description: Frontend implementation status, component index, and design notes for the webui.
---

# Frontend Plans

All phases (1–4) are **implemented**.

- `plans/frontend.md` — architecture, API endpoint table, component index, known issues
- `plans/frontend-caption-settings.md` — design decisions and Svelte 5/CM6 gotchas

## Backend Controllers

| File | Endpoints |
|------|-----------|
| `api_datasets.py` | Dataset/image CRUD, caption update, prompt preview, **add dataset (import TOML / create new)** |
| `api_captioning.py` | Start/stop/status (SSE) captioning |
| `api_envs.py` | Env CRUD + model list proxy |
| `api_templates.py` | Template CRUD + variable extraction |
| `api_configs.py` | Dataset config TOML CRUD |
| `api_export.py` | Export backends list + run export |
| `api_events.py` | Global SSE stream |

## Frontend Store Modules

| File | Purpose |
|------|---------|
| `datasetImages.ts` | Dataset/image types, CRUD, paginated store, captioning API, prompt preview, **importDataset / createDataset** |
| `envs.ts` | Env types + CRUD + model fetching |
| `templates.ts` | Template types + CRUD + `extractVariables()` |
| `captionOptions.ts` | `CaptionOptions` type (mirrors `CaptionJobOptions`) |
| `configs.ts` | Config CRUD + export API |
| `captioning.ts` | SSE event Zod schemas |
| `settings.ts` | UI settings (localStorage) |

## Key Patterns

- **CodeMirror.svelte**: Three separate `$effect` blocks (create/destroy/sync) — never combine. See `plans/frontend-caption-settings.md`. Uses `editable` prop (default `true`) — not `readonly`.
- **TomlEditor.svelte**: Wraps CodeMirror with TOML syntax, line wrapping, hidden gutters. `editable` prop (default `true`) toggles cursor visibility and write access. Used in `ImageDetail.svelte` with a View/Edit toggle for TOML extras.
- **JinjaEditor.svelte**: Wraps CodeMirror with Jinja2 syntax, variable extraction bar. Also uses `editable` prop (default `true`).
- **Svelte 5**: No pipe directives on events. No nested `<button>`. Use `<div role="button">` for clickable list items.
- **Tailwind v4**: Custom colors must be registered in `@theme { }` block, not `:root` vars.
- **SSE**: `TypedEventSource` + Zod schemas for type-safe event handling.
- **Security**: `min-release-age=14` in `.npmrc`.

## Known Issues

- ImageDetail dialog overloaded — needs tabbed/split redesign
- TOML extras editing in ImageDetail is UI-only (no backend save endpoint yet)
- No per-tile SSE updates during captioning (aggregate events only)
