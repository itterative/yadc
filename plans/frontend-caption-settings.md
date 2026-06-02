# Frontend Caption Settings — Implementation Record

All steps implemented. This file records design decisions and gotchas.

## Key Design Decisions

### Environment system
- Envs store API URL + token (RSA-encrypted via keyring) + default model name
- Tokens masked as `[REDACTED]` in API responses — never pre-filled in forms
- `POST /api/envs/<name>/models` proxies model list, normalizes OpenAI/Ollama/plain formats
- Caption settings form uses env as one-time override (doesn't auto-save back)

### Template system
- User templates override built-ins by name (deduped in listing)
- Built-in templates cannot be deleted
- Variable extraction via regex on `{{ var }}` and `{% for var in ... %}` patterns (backend + frontend)

### CodeMirror 6
- `@codemirror/lang-jinja` for Jinja2 syntax highlighting
- `@codemirror/legacy-modes/mode/toml` for readonly TOML viewer
- **CodeMirror.svelte wrapper**: Must use three separate `$effect` blocks (create, destroy, doc/extensions sync). Combining creation + prop reactivity into one effect causes "duplicate editor" bug — Svelte re-runs on prop changes → destroy → recreate → stacked DOM children.

### Caption progress
- SSE via `TypedEventSource` + Zod validation
- Grid tiles refresh only on completion (aggregate SSE events, no per-image IDs)
- 2-second delay before closing SSE on terminal state

### Prompt preview
- `PromptRenderer` extracted from `Captioner` as standalone class (`yadc/core/captioner.py`)
- Backend renders template server-side, returns `{system_prompt, user_prompt, template_context}`

### Svelte 5 gotchas
- No `onclick|stopPropagation` — use `onclick={(e) => { e.stopPropagation(); ... }}`
- No `<button>` nested in `<button>` — use `<div role="button" tabindex="0">` for list items
- Tailwind v4: custom colors need `@theme { }` block, not `:root` CSS vars, for utility generation

### Security
- `min-release-age=14` in `yadc/webui/.npmrc` blocks packages published <14 days ago
