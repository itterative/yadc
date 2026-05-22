# yadc Frontend Plan

## Reference Architecture (qwen-reranker-test)

The qwen-reranker-test project uses a **Flask + SvelteKit** split architecture:

- **Backend** (`backend/`): Flask app served via **waitress**, with two Flask blueprints:
  - `ApiBlueprint` (`/api/*`) — JSON REST endpoints + SSE event stream
  - `AppBlueprint` (`/*`) — serves the SvelteKit static build (`frontend/build/`)
- **Frontend** (`frontend/`): **SvelteKit** with `adapter-static` (SPA mode: `ssr: false`, `prerender: true`), **Svelte 5**, **Tailwind CSS v4**, **TypeScript**, **Zod** for runtime validation
- **DI**: `injector` library wires Configuration → controllers → services
- **Frontend serves from Flask**: Svelte builds to `frontend/build/`, Flask serves it directly — no separate dev server in production

### Adapted Layout for yadc

For yadc, the same architecture is placed inside the yadc package:
- **API** (`yadc/api/`) — Flask backend (was `backend/` in reference)
- **Frontend** (`yadc/webui/`) — SvelteKit frontend (was `frontend/` in reference)
- **CLI entry** (`yadc/cli_webui.py`) — `yadc webui` command to launch the server

### Key Backend Patterns
- `application.py` — creates Flask app, Injector, registers blueprints and controllers
- `controllers/` — each file is a function decorated with `@inject`, receives dependencies via injector
- `controllers/app_frontend.py` — serves `index.html`, `robots.txt`, and `/_app/*` from the static build
- `configuration.py` — `@dataclass` with all config fields (http, cors, model, paths)
- SSE events for real-time updates (ingestion status, new documents)
- CORS middleware for dev mode

### Key Frontend Patterns
- `$env/dynamic/public` for `PUBLIC_BACKEND_URL` (empty in prod, `http://localhost:5001` in dev)
- Svelte stores (`writable`) for state, custom `storable()` wrapper for localStorage persistence
- Zod schemas for SSE event validation
- Gallery-style UI: masonry grid, lazy loading via IntersectionObserver, search bar, settings dialog
- Shared components: `Dialog.svelte`, `Checkbox.svelte`, `IntersectionObserverElement.svelte`

---

## Plan for yadc Frontend

### What yadc's frontend needs to do

yadc is a **dataset captioning tool** — the frontend should let users:

1. **Browse datasets** — view images in a dataset with their current captions/extras/drafts
2. **View & edit captions** — see caption text, TOML metadata, draft files
3. **Trigger captioning** — start captioning runs with configurable options (model, template, rounds, etc.)
4. **Monitor progress** — real-time SSE updates on captioning progress (images processed, tokens used, errors)
5. **Manage configs** — view/edit dataset configs, environments, templates
6. **Export** — trigger exports to training formats

### Architecture Decisions

Replicate the same Flask + SvelteKit pattern:

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Web server | **Flask + waitress** | Matches reference project; yadc already uses Python |
| DI | **injector** | Matches reference pattern; clean separation of concerns |
| Frontend framework | **SvelteKit (adapter-static)** | Same as reference — SPA that Flask serves |
| CSS | **Tailwind CSS v4** | Same as reference |
| Type validation | **Zod** | Same as reference — validates SSE/API payloads |
| Language | **TypeScript** | Same as reference |

### Directory Structure (New Files)

```
yadc/                           # Python package root
  cli_webui.py                  # NEW — `yadc webui` CLI command (click)
  api/                          # NEW — Flask backend
    __init__.py
    application.py              # Flask app, Injector, blueprint registration
    configuration.py            # @dataclass config (http, cors, paths)
    controllers/
      __init__.py
      blueprints.py             # ApiBlueprint, AppBlueprint singletons
      app_frontend.py           # Serve SvelteKit build
      api_cors.py               # CORS middleware
      api_datasets.py           # Dataset listing, image browsing
      api_captioning.py         # Start/stop captioning, progress SSE
      api_configs.py            # Config/env/template management
      api_export.py             # Export triggers
    modules/
      __init__.py
      service.py                # Base Service class
      event_dispatcher.py       # SSE event broadcasting

  webui/                        # NEW — SvelteKit frontend
    package.json
    svelte.config.js
    vite.config.ts
    tsconfig.json
    .env                        # PUBLIC_BACKEND_URL=""
    .env.development            # PUBLIC_BACKEND_URL="http://localhost:5001"
    .npmrc
    .prettierrc
    .prettierignore
    .gitignore
    eslint.config.js
    src/
      app.html
      app.d.ts
      lib/
        index.ts
        events.ts               # TypedEventSource with Zod validation
        async.ts                # deferred, synchronized, sleep helpers
        storable.js             # localStorage-backed writable store
        stores/
          settings.ts           # UI settings (storable)
          captioning.ts         # Captioning progress state
        components/
          Dialog.svelte
          Checkbox.svelte
          IntersectionObserverElement.svelte
        icons/
          SvgSpinner.svelte
          SvgBurgerMenu.svelte
          SvgClose.svelte
          SvgPlus.svelte
      routes/
        layout.css              # Tailwind imports + theme
        +layout.svelte          # Shell with nav
        +layout.ts              # prerender=true, ssr=false
        +page.svelte            # Main dashboard
        +page.ts                # Load initial data
        DatasetBrowser.svelte   # Image grid with masonry layout
        ImageDetail.svelte      # Focused image view with caption/edit
        CaptionSettings.svelte  # Captioning config form
        SettingsDialog.svelte   # App settings
    static/
      robots.txt
```

### CLI Entry (`yadc/cli_webui.py`)

Follows the existing `cli_*.py` pattern. Registers as `yadc webui`:

```python
import click
from yadc.api.application import Application, Configuration

@click.group()
def webui():
    """Launch the yadc web UI."""
    pass

@webui.command()
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", default=7860, help="Bind port")
@click.option("--debug/--no-debug", default=False)
def serve(host, port, debug):
    """Start the web UI server."""
    configuration = Configuration(http_host=host, http_port=port)
    application = Application(configuration)
    application.run()
```

Registered in `yadc/cli.py`:
```python
from . import cli_webui
cli.add_command(cli_webui.webui)
```

### Backend API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/datasets` | List available datasets (scan for .toml configs) |
| GET | `/api/datasets/{name}/images` | List images with captions/drafts (paginated) |
| GET | `/api/datasets/{name}/images/{id}/media` | Serve image file |
| GET | `/api/datasets/{name}/images/{id}/thumbnail` | Serve/generated thumbnail |
| GET | `/api/datasets/{name}/images/{id}/caption` | Get caption text + TOML extras |
| PUT | `/api/datasets/{name}/images/{id}/caption` | Update caption/extras |
| POST | `/api/datasets/{name}/caption` | Start captioning run |
| GET | `/api/datasets/{name}/caption/status` | SSE stream for captioning progress |
| DELETE | `/api/datasets/{name}/caption` | Stop captioning run |
| GET | `/api/configs` | List user configs |
| GET | `/api/envs` | List environments |
| GET | `/api/templates` | List prompt templates |
| GET | `/api/templates/{name}` | Get template content |
| PUT | `/api/templates/{name}` | Update template |
| POST | `/api/export` | Trigger export |
| GET | `/api/events` | Global SSE event stream |

### Frontend Pages & Components

#### Main Page (`+page.svelte`)
- Dashboard with dataset selector
- Image browser (masonry grid, same pattern as GalleryContainer.svelte)
- Search/filter bar
- Settings gear icon → SettingsDialog

#### Image Detail (Dialog or Route)
- Full-size image preview
- Current caption text (editable)
- TOML metadata viewer/editor
- Draft management (view, compare, apply)
- Trigger re-caption button

#### Caption Settings Panel
- Model selection
- Template selection
- Rounds configuration
- Reasoning settings
- Start/stop controls
- Progress bar with SSE updates

#### Settings Dialog
- Environment management (api_url, model_name)
- Template management
- Cache controls

### Implementation Phases

#### Phase 1: Skeleton
1. Create `yadc/api/` with Flask app skeleton (`application.py`, `configuration.py`, `blueprints.py`, `app_frontend.py`, `api_cors.py`)
2. Create `yadc/webui/` with SvelteKit project (copy config files from reference)
3. Wire Flask to serve SvelteKit build
4. Create `yadc/cli_webui.py` with `yadc webui serve` command
5. Register `webui` group in `yadc/cli.py`
6. Verify Flask serves the frontend and API endpoints respond

#### Phase 2: Dataset Browsing API + UI
1. `yadc/api/controllers/api_datasets.py` — scan for dataset configs, list images, serve media/thumbnails
2. Frontend: `DatasetBrowser.svelte` — masonry grid with lazy loading
3. Frontend: `ImageDetail.svelte` — focused view with caption display
4. Wire up pagination with next_token pattern

#### Phase 3: Captioning Integration
1. `yadc/api/controllers/api_captioning.py` — start/stop captioning in background thread, SSE progress
2. `CaptionSettings.svelte` — config form for captioning options
3. Wire SSE events for real-time progress (images done, tokens, errors)
4. Caption display updates as images are processed

#### Phase 4: Config & Export Management
1. `yadc/api/controllers/api_configs.py` — CRUD for environments, templates, user configs
2. `yadc/api/controllers/api_export.py` — trigger exports
3. `SettingsDialog.svelte` — full settings UI
4. Export form with backend/format selection

### Configuration Integration

The backend `configuration.py` should reuse yadc's existing platformdirs paths:

```python
@dataclass
class Configuration:
    # HTTP
    http_host: str = "127.0.0.1"
    http_port: int = 7860
    http_threads: int = 4
    
    # Frontend
    app_frontend_build_path: str = "../webui/build"
    app_frontend_cache_control: str = "public, max-age=31536000, immutable"
    
    # CORS (dev mode)
    api_cors_enable: bool = True
    
    # yadc paths (from cmd/app.py)
    config_path: str = ""   # auto-resolved via platformdirs
    state_path: str = ""    # auto-resolved via platformdirs
    cache_path: str = ""    # auto-resolved via platformdirs
```

### Dev Workflow

**Frontend dev** (hot reload):
```bash
cd yadc/webui && npm run dev  # Vite dev server on :5173, proxies API to Flask
```

**Backend dev**:
```bash
uv run yadc webui serve  # Flask on :7860
```

**Production build**:
```bash
cd yadc/webui && npm run build   # outputs to yadc/webui/build/
uv run yadc webui serve          # Flask serves everything on :7860
```

### Key Differences from Reference

| Aspect | qwen-reranker-test | yadc |
|--------|-------------------|------|
| Domain | Gallery search with ML embeddings | Dataset captioning tool |
| Core operations | Search, browse media | Caption generation, editing, export |
| Real-time needs | Ingestion status, new documents | Captioning progress per image |
| Data model | Items in SQLite | Images on disk with .txt/.toml sidecars |
| Heavy computation | Model inference on search | API-based captioning (no local model) |
| Config | In-code dataclass defaults | TOML files + user configs + envs |

### Files to Copy (with Adaptation)

From the reference project, these can be largely copied and adapted:

| Reference file | yadc target | Adaptation needed |
|------|-------------|------------------|
| `backend/controllers/blueprints.py` | `yadc/api/controllers/blueprints.py` | None — identical |
| `backend/controllers/app_frontend.py` | `yadc/api/controllers/app_frontend.py` | Update build path to `webui/build` |
| `backend/controllers/api_cors.py` | `yadc/api/controllers/api_cors.py` | None — identical |
| `backend/modules/event_dispatcher.py` | `yadc/api/modules/event_dispatcher.py` | Minor — event types will differ |
| `backend/application.py` | `yadc/api/application.py` | Update imports to `yadc.api.*` |
| `backend/configuration.py` | `yadc/api/configuration.py` | Update `app_frontend_build_path` default |
| `frontend/package.json` | `yadc/webui/package.json` | Change name to "yadc-webui" |
| `frontend/svelte.config.js` | `yadc/webui/svelte.config.js` | None — identical |
| `frontend/vite.config.ts` | `yadc/webui/vite.config.ts` | None — identical |
| `frontend/tsconfig.json` | `yadc/webui/tsconfig.json` | None — identical |
| `frontend/.env*` | `yadc/webui/.env*` | Keep same pattern |
| `frontend/.npmrc` | `yadc/webui/.npmrc` | None — identical |
| `frontend/.prettierrc` | `yadc/webui/.prettierrc` | Update tailwindStylesheet path if needed |
| `frontend/.gitignore` | `yadc/webui/.gitignore` | None — identical |
| `frontend/eslint.config.js` | `yadc/webui/eslint.config.js` | None — identical |
| `frontend/src/app.html` | `yadc/webui/src/app.html` | None — identical |
| `frontend/src/app.d.ts` | `yadc/webui/src/app.d.ts` | None — identical |
| `frontend/src/lib/events.ts` | `yadc/webui/src/lib/events.ts` | None — identical |
| `frontend/src/lib/async.ts` | `yadc/webui/src/lib/async.ts` | None — identical |
| `frontend/src/lib/storable.js` | `yadc/webui/src/lib/storable.js` | None — identical |
| `frontend/src/lib/components/*.svelte` | `yadc/webui/src/lib/components/*.svelte` | None — identical (Dialog, Checkbox, IntersectionObserver) |
| `frontend/src/routes/layout.css` | `yadc/webui/src/routes/layout.css` | Update theme colors for yadc branding |
| _(new)_ | `yadc/cli_webui.py` | New file — click command group with `serve` subcommand |
