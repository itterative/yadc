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
| `events.ts` | **Self-connecting SSE store** — opens `TypedEventSource` on module load (browser), validates with Zod, pipes into `readonly` writable stores. Exports `captioningStatus`, `pendingDatasetChanges`, `clearPendingDatasetChange()` |
| `captioning.ts` | Re-export shim from `events.ts` for backward compatibility |
| `settings.ts` | UI settings (localStorage) |

## Key Patterns

- **CodeMirror.svelte**: Three separate `$effect` blocks (create/destroy/sync) — never combine. See `plans/frontend-caption-settings.md`. Uses `editable` prop (default `true`) — not `readonly`. Includes a `baseTheme` (dark surface, accent-colored selection via `color-mix(in oklch, ...)`) and a `darkHighlightStyle` (`HighlightStyle.define` + `syntaxHighlighting`) that maps all `@lezer/highlight` tags to CSS `--color-syn-*` variables defined in the Tailwind `@theme` block.
- **TomlEditor.svelte**: Wraps CodeMirror with TOML syntax, line wrapping, hidden gutters. `editable` prop (default `true`) toggles cursor visibility and write access. Used in `ImageDetail.svelte` for TOML extras editing and readonly template context display.
- **JinjaEditor.svelte**: Wraps CodeMirror with Jinja2 syntax, variable extraction bar. Also uses `editable` prop (default `true`).
- **Svelte 5**: No pipe directives on events. No nested `<button>`. Use `<div role="button">` for clickable list items.
- **Tailwind v4**: Custom colors must be registered in `@theme { }` block, not `:root` vars.
- **SSE**: `stores/events.ts` is a self-connecting store module (inspired by reference project `~/Repos/qwen-reranker-test/`). Opens `TypedEventSource` on module load, validates events with Zod, pipes into `readonly` writable stores. Components import stores directly — no SSE connection logic in page components. Per-dataset SSE (e.g. `CaptionProgress.svelte`) still creates its own `TypedEventSource`.
- **Dataset watcher**: Backend emits `DatasetChangedEvent` via SSE when filesystem changes are detected. Frontend stores these in `pendingDatasetChanges` (a `Set<string>`). Dataset browser page subscribes and shows a "Refresh" banner.
- **Security**: `min-release-age=14` in `.npmrc`.

## Known Issues

- No per-tile SSE updates during captioning (aggregate events only)
