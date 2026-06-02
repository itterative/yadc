---
name: frontend-plans
description: Location and status of frontend implementation plans for the webui.
---

# Frontend Plans

## Location

Plans live in `plans/` at the project root (not in `.pi/agent/memory/`):
- `plans/frontend.md` — master frontend plan (all phases)
- `plans/frontend-caption-settings.md` — detailed breakdown of Phase 3 items 4–6

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Skeleton (Flask + SvelteKit + DI) | ✅ Done |
| 2 | Dataset browsing API + UI | ✅ Done |
| 3 | Captioning integration | ✅ Done |
| 3.1–3.3 | Backend captioning service/controller/events | ✅ Done |
| 3.4 | Caption settings UI + supporting APIs | ✅ Done |
| 3.4 step 1 | Environment CRUD API (`api_envs.py`) | ✅ Done |
| 3.4 step 2 | Template CRUD API (`api_templates.py`) | ✅ Done |
| 3.4 step 3 | EnvManager.svelte + `envs.ts` API helpers | ✅ Done |
| 3.4 step 4 | JinjaEditor + CodeMirror setup + TomlViewer | ✅ Done (CodeMirror 6 + @codemirror/lang-jinja + @codemirror/legacy-modes/mode/toml) |
| 3.4 step 5 | CaptionSettings.svelte | ✅ Done |
| 3.4 step 6 | CaptionProgress.svelte + SSE wiring | ✅ Done |
| 4 | Config & export management | Not started |

## New API Endpoints (Phase 3)

### Environments (`api_envs.py`)
- `GET /api/envs` — list env names
- `GET /api/envs/<name>` — get env settings (token masked)
- `PUT /api/envs/<name>` — create/update env
- `DELETE /api/envs/<name>` — delete env (blocks `default`)
- `POST /api/envs/<name>/models` — proxy model list from env's API (normalizes OpenAI/Ollama/plain)

### Templates (`api_templates.py`)
- `GET /api/templates` — list user + built-in templates with source type
- `GET /api/templates/<name>` — get content + source + extracted variables
- `PUT /api/templates/<name>` — create/update user template
- `DELETE /api/templates/<name>` — delete user template (blocks built-in-only)

## Frontend Components Created (Phase 3.4)

- `EnvManager.svelte` — Dialog for env CRUD (list, create, edit, delete with confirmation)
- `JinjaEditor.svelte` — CodeMirror 6 Jinja2 editor with `@codemirror/lang-jinja`, dark Tokyo Night theme, variable extraction
- `CodeMirror.svelte` — Svelte 5 runes wrapper for CM6
- `TomlViewer.svelte` — Readonly CM6 viewer with `@codemirror/legacy-modes/mode/toml`
- `envs.ts` — API helpers + `EnvInfo` type
- `templates.ts` — API helpers + `TemplateInfo`/`TemplateListItem` types + `extractVariables()`
- `captionOptions.ts` — `CaptionOptions` type for the settings form output
- `CaptionSettings.svelte` — Main caption settings dialog (env/template/options/reasoning sections)
- `CaptionProgress.svelte` — Real-time progress bar with SSE subscription, stop button, color-coded states
- Icons: `SvgFile`, `SvgEdit`, `SvgDelete`, `SvgPlus`, `SvgRefresh`
- Backend: `datasets.py` now returns `extras_raw` (raw TOML) alongside parsed `extras`
- `datasetImages.ts` — added `startCaptioning()`, `stopCaptioning()`, `CaptioningJobInfo` type

## Editor Decision

Using **CodeMirror 6** with `@codemirror/lang-jinja` for Jinja2 templates and
`@codemirror/legacy-modes/mode/toml` for readonly TOML display.

### CodeMirror.svelte wrapper
Svelte 5 runes wrapper. Key design: three separate `$effect` blocks (create, destroy,
doc/extensions sync) to avoid the "duplicate editor" bug. Do NOT combine creation
and prop reactivity into one effect — Svelte re-runs on prop changes causing
destroy → recreate → duplicate DOM children.

### Security
`min-release-age=14` in `.npmrc` blocks packages published <14 days ago.

## Env System Backend

Envs are stored in user config TOML (`~/.config/yadc/config.toml`) under `[env.<name>]`.
Each env has: `api_url`, `api_token` (RSA-encrypted via keyring), `api_model_name`.
`cmd/envs/` module handles CRUD + encryption. The `Setting` class's `__str__` masks encrypted values as `[REDACTED]`.

## Template System Backend

User templates are `.jinja` files in `STATE_PATH/templates/` (~/.local/state/yadc/templates/).
Built-in templates live in the package at `yadc/templates/jinja/`.
`cmd/templates/` module handles CRUD. Built-in templates cannot be deleted but user templates can override them by name.
