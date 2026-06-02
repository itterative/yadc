# Frontend Plan: Caption Settings (Phase 3, items 4–6)

> Refines Phase 3 of `plans/frontend.md` — covers `CaptionSettings.svelte`,
> SSE wiring, and live caption display.

## Overview

The caption settings UI is the main control panel for starting/managing
captioning runs. It's more complex than a simple form because it needs to:

1. **Manage environments** — each env stores an API URL + optional API token +
   a default model name. Tokens are RSA-encrypted via keyring.
2. **Pick models** — user either types a model name or fetches available models
   from the selected env's API.
3. **Edit Jinja2 templates** — a text editor with Jinja2-aware features (syntax
   highlighting, variable hints).
4. **Configure simple options** — rounds, max tokens, image quality, draft
   mode, reasoning, etc.

## Existing Backend

The backend already has:

- `CaptionJobOptions` (Pydantic) — all captioning parameters including
  `api_url`, `api_token`, `api_model_name`, `env`, `prompt_template`,
  `prompt_name`, `max_tokens`, `image_quality`, `draft`, `overwrite`,
  `reasoning`, etc.
- `CaptioningService` — start/stop/status
- `POST /api/datasets/<name>/caption` — starts a job with JSON body matching
  `CaptionJobOptions`
- `DELETE /api/datasets/<name>/caption` — stops a running job
- `GET /api/datasets/<name>/caption/status` — SSE stream for progress
- `GET /api/events` — global SSE stream
- Env system: `cmd/envs` — `list_all_env()`, `load_env()`, `save_env()`,
  `get_env()`, `update_env()`, `delete_env()` — stores in user config TOML
  under `[env.<name>]` with RSA-encrypted tokens
- Template system: `cmd/templates` — `list_user_template()`,
  `load_user_template()`, `save_user_template()`, `delete_user_template()`,
  `default_template()`, `load_builtin_template()` — user templates are
  `.jinja` files in `STATE_PATH/templates/`

## New Backend API Endpoints

### Environments

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/envs` | List all env names |
| GET | `/api/envs/<name>` | Get env settings (token returned as `[REDACTED]` or masked; full token only sent on explicit create/update) |
| PUT | `/api/envs/<name>` | Create or update env (JSON: `{api_url, api_token?, api_model_name?}`) |
| DELETE | `/api/envs/<name>` | Delete an env (cannot delete `default`) |
| POST | `/api/envs/<name>/models` | Fetch available models from the env's API URL (proxied to avoid CORS) |

**Notes on `/api/envs/<name>/models`:**
- This endpoint calls the env's `api_url` (e.g. `GET {api_url}/models`) and
  returns the model list to the frontend.
- Needs a timeout and error handling for unreachable APIs.
- The response format depends on the API provider (OpenAI-compatible returns
  `{data: [{id: "model-name", ...}]}`) — we should normalize to `string[]`.
- Token is decrypted from the env and sent as `Authorization: Bearer <token>`.

### Templates

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/templates` | List all template names (user + built-in, with type flag) |
| GET | `/api/templates/<name>` | Get template content + metadata (source: user/builtin, variables used) |
| PUT | `/api/templates/<name>` | Create or update user template (JSON: `{content: string}`) |
| DELETE | `/api/templates/<name>` | Delete a user template (cannot delete built-in) |

**Template metadata:**
- `name`: template name
- `source`: `"user"` or `"builtin"`
- `content`: the Jinja2 text
- `variables`: extracted Jinja2 variable names (for editor hints) — parse with
  regex on `{{ variable }}` and `{% for var in ... %}` patterns

## Frontend Components

### `CaptionSettings.svelte`

Top-level form component — shown as a dialog/panel when user clicks
"Start Captioning" on a dataset page.

Layout (sections, top to bottom):

```
┌─────────────────────────────────────────────┐
│ Caption Settings                        [X] │
├─────────────────────────────────────────────┤
│                                             │
│  Environment  [▼ my-env        ] [Manage]   │
│                                             │
│  API URL      [https://api.example.com  ]   │
│  API Token    [•••••••••••••••••••••    ]   │
│  Model        [▼ gpt-4o-mini        ] [⟳]   │
│                                             │
│  ── Template ─────────────────────────────  │
│  Template     [▼ default         ] [New]    │
│                                             │
│  ┌─ Jinja2 Editor ────────────────────────┐ │
│  │ {% set system_prompt %}                │ │
│  │ You are a useful assistant...          │ │
│  │ {% endset %}                           │ │
│  │                                        │ │
│  │ Variables: system_prompt, user_prompt  │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  ── Options ───────────────────────────────  │
│  Max Tokens   [512    ]                     │
│  Image Quality [▼ auto ]                    │
│  Draft        [        ] (draft name)       │
│  Overwrite    [☐]                           │
│  Rounds       [1]                           │
│                                             │
│  ── Reasoning ─────────────────────────────  │
│  Enable       [☐]                           │
│  Effort       [▼ low ]                      │
│                                             │
│                    [Cancel] [Start Caption]  │
└─────────────────────────────────────────────┘
```

**Behavior:**
- Selecting an env auto-fills URL, token, model from that env's settings.
- Changing URL/token/model manually does NOT modify the saved env (one-time
  override for this run). Optionally show a "Save to env" checkbox/button.
- Model dropdown: populated via `POST /api/envs/<name>/models` when user
  clicks refresh [⟳] or changes the env. Falls back to free-text input if
  the API call fails.
- Template dropdown: shows user + built-in templates. Selecting one loads it
  into the editor. "New" opens a name input.
- The Jinja2 editor is always visible (not behind a toggle) — the template
  dropdown and editor are two views of the same data.

### `EnvManager.svelte`

Dialog for CRUD on environments. Simple table:

```
┌──────────────────────────────────────┐
│ Manage Environments             [X]  │
├──────────────────────────────────────┤
│  Name    URL                  Actions │
│  default https://api.openai.com [✏][🗑]│
│  local   http://localhost:11434 [✏][🗑]│
│                                      │
│  [+ New Environment]                 │
└──────────────────────────────────────┘
```

Each env row shows name + URL (token masked). Edit opens inline fields or a
sub-dialog. Delete confirms. Cannot delete `default`.

### `JinjaEditor.svelte`

Text editor with Jinja2 awareness:

- **Syntax highlighting**: Use a lightweight approach — either:
  - (a) **CodeMirror 6** with a custom Jinja2/HTML-ish language mode, or
  - (b) **Monaco Editor** (heavy, probably overkill), or
  - (c) A simple `<textarea>` with a separate highlighted preview panel.
  - **Recommendation: CodeMirror 6** — ~40KB gzipped, great Svelte integration,
    easy to add custom highlighting. Has `@codemirror/lang-html` as a base,
    plus custom Jinja2 delimiters (`{{ }}`, `{% %}`, `{# #}`).
- **Variable hints**: Parse template text for `{{ var }}` references and show a
  small "Available variables" sidebar or tooltip.
- **Monospace font**, line numbers, basic keybindings.

### `CaptionProgress.svelte`

Real-time progress display during captioning:

```
┌──────────────────────────────────────┐
│ Captioning: my_dataset               │
│ ████████████░░░░░░  245 / 500        │
│ Errors: 2  ·  Tokens: 12,450         │
│                       [Stop Caption]  │
└──────────────────────────────────────┘
```

- Subscribes to SSE `CaptioningStatusEvent` for the active dataset.
- Shows progress bar, processed/total, error count.
- "Stop" button calls `DELETE /api/datasets/<name>/caption`.

### Caption display updates in `DatasetBrowser` / `ImageDetail`

- When SSE reports a new image processed, the masonry grid should update the
  image tile's caption preview (if visible).
- The `ImageDetail` dialog should show a "Caption updated" indicator if the
  caption changes while the dialog is open.

## Implementation Steps

### Step 1: Environment API endpoints ✅ DONE

Created `yadc/api/controllers/api_envs.py`:

1. `GET /api/envs` — list env names via `cmd_envs.list_all_env()`
2. `GET /api/envs/<name>` — return env settings (token masked via `Setting.__str__`)
3. `PUT /api/envs/<name>` — create/update env (saves only provided keys)
4. `DELETE /api/envs/<name>` — delete env (blocks `default`)
5. `POST /api/envs/<name>/models` — proxy model list from env's API, normalizes OpenAI/Ollama/plain list formats

Auto-discovered by `discovery.discover_controllers()`.

### Step 2: Template API endpoints ✅ DONE

Created `yadc/api/controllers/api_templates.py`:

1. `GET /api/templates` — list user + built-in template names with `source` type (deduplicates user overrides of builtins)
2. `GET /api/templates/<name>` — return content + `source` + `variables` (extracted via regex)
3. `PUT /api/templates/<name>` — create/update user template (validates content is string)
4. `DELETE /api/templates/<name>` — delete user template (blocks built-in-only names)
5. `_extract_variables()` — regex-based Jinja2 variable extraction for editor hints

Auto-discovered by `discovery.discover_controllers()`.

### Step 3: Frontend — EnvManager ✅ DONE

1. `EnvManager.svelte` — env CRUD dialog (list/create/edit/delete, token masking, delete confirmation)
2. `envs.ts` — types (`EnvInfo`) + API helpers (`fetchEnvs`, `fetchEnv`, `saveEnv`, `deleteEnv`, `fetchModels`)
3. New icons: `SvgFile`, `SvgEdit`, `SvgDelete`, `SvgPlus`, `SvgRefresh`

### Step 4: Frontend — JinjaEditor ✅ DONE (textarea-based, no CodeMirror)

1. `JinjaEditor.svelte` — textarea-based editor with:
   - Monospace font, tab key inserts 2 spaces
   - Reactive `extractVariables()` showing detected Jinja2 variables as tags below the editor
   - `readonly` mode support
   - `value` bindable + `onchange` callback
2. `templates.ts` — types (`TemplateInfo`, `TemplateListItem`) + API helpers + `extractVariables()` frontend utility

**Note:** CodeMirror 6 is NOT installed yet. The plan originally called for CM6 but the user wants to
evaluate leaner alternatives first. Current version is a plain `<textarea>` with variable extraction.
See "Editor Options" below for the evaluation.

#### Editor Options (CodeMirror 6 alternatives evaluation)

**CodeMirror 6** (v6.43.0) — the original plan choice:
- ~2MB unpacked for core (`@codemirror/view` 1.2MB + `@codemirror/state` 433KB + `@codemirror/language` 310KB)
- Full-featured editor: line numbers, syntax highlighting, code folding, search, autocomplete
- Needs `@codemirror/lang-html` + custom Jinja2 delimiter mode
- 7 transitive deps in `codemirror` package alone

**PrismJS** (v1.30.0) — syntax highlighting only, not an editor:
- ~2MB unpacked (includes all languages; ~5KB for a custom Jinja-like grammar)
- Zero dependencies
- Read-only highlighting — could be used as an overlay on a transparent textarea
- Would need manual sync between textarea scroll and highlight layer

**highlight.js** (v11.11.1) — syntax highlighting only:
- ~5.4MB unpacked (all languages bundled; tree-shakeable to ~50KB with specific languages)
- Zero dependencies
- Same overlay approach as PrismJS

**CodeFlask** (v1.4.1) — tiny code editor built on PrismJS:
- ~55KB unpacked, depends on PrismJS
- Provides a real editor (not just highlighting) — wraps textarea + PrismJS highlight
- Very lightweight, but limited customization

**Textarea + highlighted preview** — pure approach:
- Zero dependencies
- `<textarea>` for editing, `<pre><code>` panel showing highlighted output
- Could use a simple regex-based highlighter for Jinja2 delimiters (`{{ }}`, `{% %}`, `{# #}`)
- Simplest approach, good enough for Jinja2 templates which are mostly prose with scattered tags

**Recommendation to revisit:** For a Jinja2 template editor, a full code editor like CodeMirror is
arguably overkill. The templates are typically short (10-30 lines) and mostly prose text with a few
Jinja2 blocks. The textarea + preview or PrismJS overlay approaches would be lighter. If we do
want a proper editor experience (line numbers, bracket matching, etc.), CodeMirror 6 is the right
choice despite the weight.

### Step 5: Frontend — CaptionSettings

1. `CaptionSettings.svelte` — main form component
2. `templates.ts` — types + API helpers for template endpoints
3. Wire env selector → auto-fill → model fetch flow
4. Wire template selector → editor load/save flow

### Step 6: Frontend — CaptionProgress + SSE wiring

1. `CaptionProgress.svelte` — progress bar with SSE subscription
2. Wire into `DatasetBrowser` — show progress when captioning is active
3. Update `ImageDetail` and grid tiles on processed-image events
4. Re-use `CaptioningStatusEvent` Zod schema from `stores/captioning.ts`

## Dependencies to Add

### Python
None — all backend logic exists in `cmd/envs` and `cmd/templates`.

### npm (`yadc/webui/`)
- ~~`codemirror` + related packages~~ — **ON HOLD**, evaluating lighter alternatives
  (see Step 4 editor options evaluation above)

## Open Questions

1. **Model list normalization** — Different API providers return different
   formats. OpenAI-compatible returns `{data: [{id: "..."}]}`. Ollama returns
   something different. Should we try to normalize, or just return the raw
   response and let the frontend show `id` fields?

2. **Template variables** — Should the backend parse Jinja2 variables and
   return them in the template metadata, or should the frontend handle this?
   (Frontend seems more natural since it's a display concern.)

3. **CodeMirror vs simpler approach** — Is CodeMirror 6 worth the dependency
   weight for a template editor? A `<textarea>` with a separate syntax-
   highlighted preview might be lighter. Decision: start with CodeMirror —
   it provides a much better editing experience and ~40KB is acceptable.

4. **Env auto-save** — When user changes URL/token/model in the caption
   settings form, should we offer to save back to the env? Or keep it strictly
   as a one-time override? (Current plan: one-time override with optional
   "Save to env" button.)
