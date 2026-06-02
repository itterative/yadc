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
| 3 | Captioning integration | In progress |
| 3.1–3.3 | Backend captioning service/controller/events | ✅ Done |
| 3.4 | Caption settings UI + supporting APIs | In progress |
| 3.4 step 1 | Environment CRUD API (`api_envs.py`) | ✅ Done |
| 3.4 step 2 | Template CRUD API (`api_templates.py`) | ✅ Done |
| 3.4 step 3 | EnvManager.svelte + `envs.ts` API helpers | ✅ Done |
| 3.4 step 4 | JinjaEditor.svelte + `templates.ts` API helpers | ✅ Done (textarea-based, no CodeMirror) |
| 3.4 step 5 | CaptionSettings.svelte | Not started |
| 3.4 step 6 | CaptionProgress.svelte + SSE wiring | Not started |
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
- `JinjaEditor.svelte` — Textarea-based Jinja2 template editor with variable extraction display
- `envs.ts` — API helpers + `EnvInfo` type
- `templates.ts` — API helpers + `TemplateInfo`/`TemplateListItem` types + `extractVariables()`
- Icons: `SvgFile`, `SvgEdit`, `SvgDelete`, `SvgPlus`, `SvgRefresh`

## Editor Decision

CodeMirror 6 was the original plan. Currently using a plain textarea with variable hints.
The plan doc has a detailed comparison of CM6 vs PrismJS vs highlight.js vs CodeFlask vs textarea+preview.
User wants to evaluate leaner options before committing to CM6.

## Env System Backend

Envs are stored in user config TOML (`~/.config/yadc/config.toml`) under `[env.<name>]`.
Each env has: `api_url`, `api_token` (RSA-encrypted via keyring), `api_model_name`.
`cmd/envs/` module handles CRUD + encryption. The `Setting` class's `__str__` masks encrypted values as `[REDACTED]`.

## Template System Backend

User templates are `.jinja` files in `STATE_PATH/templates/` (~/.local/state/yadc/templates/).
Built-in templates live in the package at `yadc/templates/jinja/`.
`cmd/templates/` module handles CRUD. Built-in templates cannot be deleted but user templates can override them by name.
