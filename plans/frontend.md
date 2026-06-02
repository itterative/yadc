# yadc Frontend Plan

All phases are **implemented**.

## Architecture

| Layer | Technology |
|-------|-----------|
| Web server | Flask + waitress |
| DI | injector + auto-discovery (`discover_services`/`discover_controllers`) |
| Frontend | SvelteKit (adapter-static, hash routing, Svelte 5, TypeScript) |
| CSS | Tailwind CSS v4 (Tokyo Night dark theme) |
| Validation | Zod (SSE/API payloads on frontend), Pydantic (API on backend) |
| Editor | CodeMirror 6 (`@codemirror/lang-jinja`, `@codemirror/legacy-modes/mode/toml`) |

### Layout

- **API** (`yadc/api/`) — Flask backend
- **Frontend** (`yadc/webui/`) — SvelteKit SPA
- **CLI** (`yadc/cli_webui.py`) — `yadc webui serve` entry point

### How to run

```bash
# Frontend dev (hot reload on :5173)
cd yadc/webui && npm run dev

# Backend dev (Flask on :7860)
uv run yadc webui serve

# Production build + serve
cd yadc/webui && npm run build
uv run yadc webui serve   # serves everything on :7860
```

## Backend API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/datasets` | List datasets |
| POST | `/api/datasets` | Import or create dataset |
| DELETE | `/api/datasets/{name}` | Delete dataset + state dir |
| POST | `/api/datasets/{name}/rescan` | Force rescan |
| GET | `/api/datasets/{name}/images` | List images (paginated, `after_id` cursor) |
| GET | `/api/datasets/{name}/images/{id}/media` | Serve image file |
| GET | `/api/datasets/{name}/images/{id}/thumbnail` | Serve thumbnail |
| GET | `/api/datasets/{name}/images/{id}/caption` | Get caption + extras + drafts |
| PUT | `/api/datasets/{name}/images/{id}/caption` | Update caption |
| POST | `/api/datasets/{name}/images/{id}/preview-prompt` | Render Jinja2 prompt preview |
| POST | `/api/datasets/{name}/caption` | Start captioning run |
| GET | `/api/datasets/{name}/caption/status` | SSE progress stream |
| DELETE | `/api/datasets/{name}/caption` | Stop captioning run |
| GET | `/api/envs` | List env names |
| GET | `/api/envs/{name}` | Get env settings (token masked) |
| PUT | `/api/envs/{name}` | Create/update env |
| DELETE | `/api/envs/{name}` | Delete env (blocks `default`) |
| POST | `/api/envs/{name}/models` | Proxy model list from env's API |
| GET | `/api/templates` | List templates (user + builtin) |
| GET | `/api/templates/{name}` | Get template content + metadata |
| PUT | `/api/templates/{name}` | Create/update user template |
| DELETE | `/api/templates/{name}` | Delete user template |
| GET | `/api/configs` | List dataset configs |
| GET | `/api/configs/{name}` | Get config TOML (raw + parsed) |
| PUT | `/api/configs/{name}` | Update config (validates + rescan) |
| DELETE | `/api/configs/{name}` | Delete config + unregister |
| GET | `/api/export/backends` | List export backends + formats |
| POST | `/api/export` | Run export |
| GET | `/api/events` | Global SSE event stream |

## Frontend Components

### Pages
- `+page.svelte` — Dataset listing (grid of cards with stats)
- `datasets/[name]/+page.svelte` — Dataset browser (masonry grid, caption button, progress)

### Dialogs
- `CaptionSettings.svelte` — Captioning config (env → model → template → options → reasoning)
- `CaptionProgress.svelte` — Real-time progress (SSE, progress bar, stop button)
- `SettingsDialog.svelte` — Tabbed: Configs (TOML editor) / Templates (Jinja2 editor) / Environments
- `ExportDialog.svelte` — Export form (dataset, backend, format, source, output)
- `ImageDetail.svelte` — Image detail (caption edit, TOML extras, drafts, prompt preview)
- `EnvManager.svelte` — Env CRUD

### Shared
- `DatasetBrowser.svelte` / `DatasetImage.svelte` — Masonry grid with lazy loading
- `CodeMirror.svelte` — Svelte 5 CM6 wrapper
- `JinjaEditor.svelte` / `TomlViewer.svelte` — CM6 editors
- `Dialog.svelte` / `Checkbox.svelte` / `IntersectionObserverElement.svelte`

### Stores (`src/lib/stores/`)
- `datasetImages.ts` — Dataset/image CRUD, captioning API, prompt preview
- `envs.ts` — Env CRUD + model fetching
- `templates.ts` — Template CRUD + variable extraction
- `captionOptions.ts` — CaptionOptions type
- `configs.ts` — Config CRUD + export API
- `captioning.ts` — SSE event schemas
- `settings.ts` — UI settings (localStorage-backed)

## Known Issues / TODOs

- **ImageDetail dialog UX** — overloaded with caption, TOML, drafts, prompt preview all stacked vertically. Needs tabbed/split redesign.
- **Per-tile SSE updates** — `CaptioningStatusEvent` only has aggregate counts, not per-image IDs. Grid does full reload on completion.
- **Tailwind v4 content detection** — needed `@theme` block (not `:root`) for custom color utilities. `@source` directives no longer needed after fix.
