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

### Step 4: Frontend — JinjaEditor + CodeMirror setup ✅ DONE

1. `JinjaEditor.svelte` — CodeMirror 6 editor with:
   - `@codemirror/lang-jinja` for Jinja2 syntax highlighting + autocomplete
   - Dark Tokyo Night theme matching the app palette
   - Line wrapping, monospace font
   - Reactive `extractVariables()` showing detected Jinja2 variables as tags below the editor
   - `readonly` mode support
   - `value` bindable + `onchange` callback
2. `templates.ts` — types (`TemplateInfo`, `TemplateListItem`) + API helpers + `extractVariables()` frontend utility
3. `CodeMirror.svelte` — Svelte 5 runes wrapper for CM6 (doc sync, extension reconfig, readonly)
4. `TomlViewer.svelte` — readonly CM6 viewer with `@codemirror/legacy-modes/mode/toml` via `StreamLanguage`
5. Backend `datasets.py` now returns `extras_raw` (raw TOML string) alongside parsed `extras`
6. `ImageDetail.svelte` uses `TomlViewer` for TOML extras display

#### npm packages installed
- `codemirror`, `@codemirror/view`, `@codemirror/state`, `@codemirror/language`, `@codemirror/commands`
- `@codemirror/lang-jinja` — official Jinja2 language support
- `@codemirror/legacy-modes` — TOML mode for readonly viewer

#### Security: `min-release-age=14` in `.npmrc`
Blocks installing any package version published less than 14 days ago.

#### CodeMirror.svelte design notes
The wrapper uses three separate `$effect` blocks to avoid the "duplicate editor" bug:
1. **Create** — fires once when DOM is ready, creates an empty `EditorView`
2. **Destroy** — cleanup on unmount
3. **Doc sync** + **Extensions sync** — separate effects that update the existing view

Do NOT combine creation + prop reactivity into one `$effect` — Svelte may re-run it on prop changes,
causing destroy → recreate → duplicate DOM children.

### Step 5: Frontend — CaptionSettings ✅ DONE

1. `CaptionSettings.svelte` — main form component with:
   - **Environment section**: dropdown selector → auto-fills URL/token/model from env, "Manage…" opens EnvManager sub-dialog
   - **Model section**: dropdown (if models fetched) or text input, refresh button fetches via `POST /api/envs/<name>/models`
   - **Template section**: dropdown (user + built-in) → loads into JinjaEditor, "New" creates inline, "Save" persists changes
   - **Options section**: max tokens, image quality, draft name, rounds, overwrite checkbox
   - **Reasoning section**: enable checkbox → thinking effort dropdown (conditional)
   - "Start Captioning" button assembles `CaptionOptions` and calls `onstart` callback
2. `captionOptions.ts` — `CaptionOptions` type (mirrors backend `CaptionJobOptions`)
3. Wired into `datasets/[name]/+page.svelte` — "Caption…" button in header opens the dialog
4. Actual captioning POST call intentionally skipped (TODO in `handleStartCaptioning`, will be done in Step 6)

### Step 6: Frontend — CaptionProgress + SSE wiring ✅ DONE

1. `CaptionProgress.svelte` — inline progress component with:
   - SSE subscription to `GET /api/datasets/<name>/caption/status` via `TypedEventSource` + Zod validation
   - Progress bar (processed/total + percentage) with color-coded states (running=accent, stopping=yellow, done=success, error=error)
   - Status header with spinner (running/stopping), checkmark (done), X (error)
   - Error detail display
   - Stop button → `DELETE /api/datasets/<name>/caption`
   - Auto-close SSE + fire `ondone` callback 2s after terminal state
2. `startCaptioning()` and `stopCaptioning()` API helpers in `datasetImages.ts`
3. Wired into `datasets/[name]/+page.svelte`:
   - `handleStartCaptioning` POSTs to caption API, sets `isCaptioning = true`
   - `CaptionProgress` shown when `isCaptioning` is true
   - "Caption…" button disabled while captioning is active
   - `handleCaptioningDone` refreshes image list + dataset stats on completion
   - Error display for failed start attempts

**Note on per-tile updates**: The current `CaptioningStatusEvent` only has aggregate counts (processed/total/errors), not per-image IDs. Grid tiles update via full refresh when captioning completes. Per-image live updates would require backend changes to emit image-level events.

### Prompt Preview ✅ DONE

A "Preview Prompt" feature on each `ImageDetail` dialog that renders a Jinja2 template
against the image's data and shows the resulting system/user prompts + template context.

**Backend:**
- `POST /api/datasets/<name>/images/<id>/preview-prompt` — accepts `{template: "..."}` or `{template_name: "..."}`,
  resolves the template, renders via `Captioner.prompts_from_image()`, returns `{system_prompt, user_prompt, template_context}`
- `DatasetService.preview_prompt()` — loads image + TOML extras + drafts, builds `DatasetImage`, renders with `PromptRenderer`
- `PromptRenderer` (`yadc/core/captioner.py`) — extracted from `Captioner`, pure Jinja2 rendering with no API dependencies

**Frontend:**
- `fetchPromptPreview()` API helper in `datasetImages.ts` — POSTs to the endpoint, returns `PromptPreview` type
- `ImageDetail.svelte` — collapsible "Preview Prompt" section with template dropdown (populated from templates API) + "Render" button,
  displays system prompt, user prompt, and a `<details>` with the raw template context variables
- Filters noise from context display (hides `caption_suffix`, `toml_suffix`, `history_suffix`)

## Dependencies Added

### Python
None — all backend logic exists in `cmd/envs` and `cmd/templates`.

### npm (`yadc/webui/`)
- `codemirror`, `@codemirror/view`, `@codemirror/state`, `@codemirror/language`, `@codemirror/commands` — core CM6
- `@codemirror/lang-jinja` — Jinja2 syntax highlighting + autocomplete
- `@codemirror/legacy-modes` — TOML stream language for readonly viewer

## Open Questions

1. **Model list normalization** — Different API providers return different
   formats. OpenAI-compatible returns `{data: [{id: "..."}]}`. Ollama returns
   something different. Should we try to normalize, or just return the raw
   response and let the frontend show `id` fields?

2. **Template variables** — Should the backend parse Jinja2 variables and
   return them in the template metadata, or should the frontend handle this?
   (Frontend seems more natural since it's a display concern.)

3. ~~CodeMirror vs simpler approach~~ — **Resolved**: Using CodeMirror 6 with `@codemirror/lang-jinja`.
   Official package exists with Jinja2 support out of the box.

4. **Env auto-save** — When user changes URL/token/model in the caption
   settings form, should we offer to save back to the env? Or keep it strictly
   as a one-time override? (Current plan: one-time override with optional
   "Save to env" button.)

## TODO: ImageDetail dialog UX refinement

The `ImageDetail` dialog is getting overloaded — caption, TOML extras, drafts,
prompt preview, status badges all stacked vertically in a scrollable panel.
Needs a redesign to avoid excessive scrolling. Possible approaches:

- **Tabbed panel** (Caption | Metadata | Preview) — each section gets its own tab
- **Sidebar + detail split** — image on left, tabs on right
- **Collapsible sections** with smart defaults (e.g., only one expanded at a time)
- **Sub-dialogs** — "Preview Prompt" and "TOML Extras" as separate dialogs
  opened from buttons in the main view

Should be revisited after Phase 3 is complete, once all the content that needs
to live in ImageDetail is finalized.
