# yadc Frontend Plan

## Reference Implementation

`~/Repos/qwen-reranker-test/` — a **Flask + SvelteKit** split architecture:

- **Backend** (`backend/`): Flask app served via **waitress**, with two Flask blueprints:
  - `ApiBlueprint` (`/api/*`) — JSON REST endpoints + SSE event stream
  - `AppBlueprint` (`/*`) — serves the SvelteKit static build (`frontend/build/`)
- **Frontend** (`frontend/`): **SvelteKit** with `adapter-static` (SPA mode: `ssr: false`, `prerender: true`), **Svelte 5**, **Tailwind CSS v4**, **TypeScript**, **Zod** for runtime validation
- **DI**: `injector` library wires Configuration → controllers → services
- **Frontend serves from Flask**: Svelte builds to `frontend/build/`, Flask serves it directly — no separate dev server in production

### Key Backend Patterns (reference)
- `application.py` — creates Flask app, Injector, registers blueprints and controllers
- `controllers/` — each file is a function decorated with `@inject`, receives dependencies via injector
- `controllers/app_frontend.py` — serves `index.html`, `robots.txt`, and `/_app/*` from the static build
- `configuration.py` — `@dataclass` with all config fields (http, cors, model, paths)
- SSE events for real-time updates (ingestion status, new documents)
- CORS middleware for dev mode

### Key Frontend Patterns (reference)
- `$env/dynamic/public` for `PUBLIC_BACKEND_URL` (empty in prod, `http://localhost:5001` in dev)
- Svelte stores (`writable`) for state, custom `storable()` wrapper for localStorage persistence
- Zod schemas for SSE event validation
- Gallery-style UI: masonry grid, lazy loading via IntersectionObserver, search bar, settings dialog
- Shared components: `Dialog.svelte`, `Checkbox.svelte`, `IntersectionObserverElement.svelte`

---

## What yadc's frontend needs to do

yadc is a **dataset captioning tool** — the frontend should let users:

1. **Browse datasets** — view images in a dataset with their current captions/extras/drafts
2. **View & edit captions** — see caption text, TOML metadata, draft files
3. **Trigger captioning** — start captioning runs with configurable options (model, template, rounds, etc.)
4. **Monitor progress** — real-time SSE updates on captioning progress (images processed, tokens used, errors)
5. **Manage configs** — view/edit dataset configs, environments, templates
6. **Export** — trigger exports to training formats

---

## Architecture Decisions

Replicate the same Flask + SvelteKit pattern, adapted for yadc's package structure:

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Web server | **Flask + waitress** | Matches reference project; yadc already uses Python |
| DI | **injector** (Phase 2+) | Matches reference pattern; clean separation of concerns |
| Frontend framework | **SvelteKit (adapter-static)** | Same as reference — SPA that Flask serves |
| CSS | **Tailwind CSS v4** | Same as reference |
| Type validation | **Zod** | Same as reference — validates SSE/API payloads |
| Language | **TypeScript** | Same as reference |

### Adapted Layout

- **API** (`yadc/api/`) — Flask backend (was `backend/` in reference)
- **Frontend** (`yadc/webui/`) — SvelteKit frontend (was `frontend/` in reference)
- **CLI entry** (`yadc/cli_webui.py`) — `yadc webui` command to launch the server

---

## Implementation Phases

### Phase 1: Skeleton ✅ DONE

All files created and verified. Flask serves the SvelteKit build, API stubs respond.

#### What was implemented

**Python dependencies added** (`pyproject.toml`):
- `flask>=3.1.0`, `waitress>=3.0.0`, `injector>=0.22.0`

**Backend** (`yadc/api/`):
```
yadc/api/
  __init__.py
  application.py          — Flask app factory, auto-discovers services/controllers, wires DI, runs via waitress
  configuration.py        — @dataclass config (http, cors, yadc paths from platformdirs)
  discovery.py            — Auto-discovery of Service subclasses and @controller functions via package scanning
  json_utils.py            — DataclassJSONEncoder + jsonify_dataclass (shared JSON utility)
  controllers/
    __init__.py            — @controller decorator (marks functions for auto-discovery + applies @inject)
    blueprints.py         — ApiBlueprint (/api/*), AppBlueprint (/*) singletons
    app_frontend.py       — Serves SvelteKit build, SPA fallback, helpful message if not built
    api_datasets.py       — Dataset/image CRUD endpoints (wired to DatasetService)
    api_captioning.py     — Stub: POST/DELETE /api/datasets/<name>/caption, GET .../status (SSE)
    api_events.py         — GET /api/events (global SSE, uses DataclassJSONEncoder)
  events.py                — Event base class + dataclass events (PingEvent, CaptioningStatusEvent)
  modules/
    __init__.py
    service.py             — Base Service class (marker for DI auto-discovery)
    cors_middleware.py     — CORSMiddleware service (origin-based CORS, registered on ApiBlueprint)
    logging_factory.py     — Logger factory, per-module named loggers
    event_dispatcher.py    — EventDispatcher, subscribe/dispatch pattern, @event_handler decorator
    job_scheduler.py       — JobScheduler, manages daemon threads + periodic cleanup
    sse_events.py          — SSEEvents, SSE push/receive with condition var, auto-ping
    db_migrations.py       — Step-based SQLite migration runner
    db_connection_factory.py — SQLite WAL, foreign keys, background init
```

**CLI** (`yadc/cli_webui.py`):
- `yadc webui serve` command with `--host`, `--port`, `--threads`, `--cors` options
- Registered as `webui` group in `yadc/cli.py`

**Frontend** (`yadc/webui/`):
```
yadc/webui/
  package.json            — SvelteKit + Svelte 5 + Tailwind CSS v4 + Zod
  svelte.config.js        — adapter-static with SPA fallback
  vite.config.ts          — tailwindcss + sveltekit plugins
  tsconfig.json
  .npmrc
  .env                    — API_BASE="" (production: same-origin)
  .env.development        — API_BASE="http://localhost:7860" (dev: separate servers)
  .gitignore
  src/
    app.html
    app.d.ts
    lib/
      index.ts
      api.ts              — API_BASE export (replaces $env/dynamic/public approach)
      events.ts           — TypedEventSource with Zod validation
      async.ts            — deferred, sleep helpers
      storable.js         — localStorage-backed writable store
      stores/
        settings.ts       — UI settings (storable)
        captioning.ts     — Captioning progress state
      components/
        Dialog.svelte
        Checkbox.svelte
    routes/
      layout.css          — Tailwind imports + dark theme (Tokyo Night palette)
      +layout.ts          — prerender=true, ssr=false
      +layout.svelte      — Shell with nav bar
      +page.svelte        — Main dashboard (fetches /api/datasets)
  static/
    robots.txt
```

#### Deviations from original plan

| Planned | Actual | Reason |
|---------|--------|--------|
| `$env/dynamic/public` for backend URL | `$lib/api.ts` with `API_BASE` constant | SvelteKit's `adapter-static` doesn't expose dynamic env vars at build time without more setup |
| `injector` DI in Phase 1 | ~~Simple module-level wiring~~ Full injector DI | Done — `Application(Module)` with `configure_services()`, `configure_controllers()`, `get_bindings()` |
| `@inject` decorator on controllers | ~~Functions registered directly~~ `@controller` decorator | Done — controllers use `@controller` (in `controllers/__init__.py`) for auto-discovery + DI |
| Full CORS with origin reflection | ~~Simple `Access-Control-Allow-Origin: *`~~ Origin-based CORS | Done — `CORSMiddleware` service, registered on `ApiBlueprint` |
| Manual service/controller lists in `application.py` | ~~Hardcoded lists~~ Auto-discovery | Done — `discover_services()` / `discover_controllers()` scan packages; `application.py` has no hardcoded lists |
| `@inject` / `@singleton` on Service classes | ~~Per-class decorators~~ Programmatic binding | Done — `Application.configure()` binds all services with `inject(cls)` + `scope=singleton`; Service classes are plain |
| `api_cors.py` controller | ~~Controller with endpoints~~ `CORSMiddleware` service | CORS is infrastructure, not a controller — moved to `modules/cors_middleware.py` |
| `app_frontend.py` serves `/_app/*` only | Serves `/_app/*` + SPA fallback via 404 handler | Matches reference more closely now |
| `.prettierrc`, `.prettierignore`, `eslint.config.js` | ✅ Added | Done — matches reference |
| Icon components (`SvgSpinner`, etc.) | ✅ Added (SvgSpinner, SvgClose, SvgImage, SvgLogout) | Done |

#### How to run

```bash
# Frontend dev (hot reload on :5173)
cd yadc/webui && npm run dev

# Backend dev (Flask on :7860)
uv run yadc webui serve

# Production build + serve
cd yadc/webui && npm run build
uv run yadc webui serve   # serves everything on :7860
```

#### Reference files to copy from (Phase 2+)

| Reference file (`~/Repos/qwen-reranker-test/`) | yadc target | Notes |
|------|-------------|-------|
| `backend/controllers/blueprints.py` | `yadc/api/controllers/blueprints.py` | Done — simplified version (no injector) |
| `backend/controllers/app_frontend.py` | `yadc/api/controllers/app_frontend.py` | Done — adapted for yadc paths |
| `backend/controllers/api_cors.py` | `yadc/api/modules/cors_middleware.py` | Done — CORS as a Service (infrastructure, not a controller) |
| `backend/modules/event_dispatcher.py` | `yadc/api/modules/event_dispatcher.py` | Done — subscribe/dispatch, @event_handler decorator |
| `backend/modules/service.py` | `yadc/api/modules/service.py` | Done — base Service class |
| `backend/modules/db_migrations.py` | `yadc/api/modules/db_migrations.py` | Done — migration runner with settings + datasets tables |
| `backend/modules/db_connection_factory.py` | `yadc/api/modules/db_connection_factory.py` | Done — SQLite WAL, foreign keys, background init |
| `backend/application.py` | `yadc/api/application.py` | Done — injector DI wired up |
| `backend/configuration.py` | `yadc/api/configuration.py` | Done — yadc-specific fields |
| `frontend/src/lib/events.ts` | `yadc/webui/src/lib/events.ts` | Done |
| `frontend/src/lib/async.ts` | `yadc/webui/src/lib/async.ts` | Done |
| `frontend/src/lib/storable.js` | `yadc/webui/src/lib/storable.js` | Done |
| `frontend/src/lib/components/Dialog.svelte` | `yadc/webui/src/lib/components/Dialog.svelte` | Done — adapted for Svelte 5 props API |
| `frontend/src/lib/components/Checkbox.svelte` | `yadc/webui/src/lib/components/Checkbox.svelte` | Done — adapted for Svelte 5 |
| `frontend/src/lib/components/IntersectionObserverElement.svelte` | `yadc/webui/src/lib/components/IntersectionObserverElement.svelte` | Done — copied from reference |
| `frontend/src/lib/icons/*.svelte` | `yadc/webui/src/lib/icons/*.svelte` | Partial — SvgSpinner, SvgClose, SvgImage, SvgLogout done |
| `frontend/src/routes/layout.css` | `yadc/webui/src/routes/layout.css` | Done — yadc dark theme |

---

### Phase 2: Dataset Browsing API + UI

1. ~~**Wire up injector**~~ ✅ — `injector`-based DI with auto-discovery; services bound programmatically (no `@inject`/`@singleton` decorators), controllers use `@controller` decorator
2. ~~**Database layer**~~ ✅ — `DBMigrations` (step-based SQLite migrations) + `DBConnectionFactory` (WAL, foreign keys, background init); tables: `properties`, `settings`, `datasets`, `dataset_images`
3. ~~`yadc/api/services/` — dataset & settings service/repos~~ ✅ — `DatasetService` (filesystem scanning + DB indexing + paginated queries + caption read/write) and `SettingsService` (KV store over `settings` table). Services live in `yadc/api/services/` package, auto-discovered alongside `modules/`.
4. ~~`yadc/api/controllers/api_datasets.py` — wire up controller stubs to use `DatasetService`~~ ✅ — Full implementation: `GET /datasets`, `GET /datasets/<name>/images` (paginated), `GET .../media`, `GET .../thumbnail` (cached in `<cache_path>/thumbnails/` as WebP), `GET .../caption`, `PUT .../caption`. Extracted shared `DataclassJSONEncoder` + `jsonify_dataclass` into `yadc/api/json_utils.py` (also used by `api_events.py`).
5. ~~`yadc/webui/src/lib/components/IntersectionObserverElement.svelte`~~ ✅ — copied from reference
6. Frontend: `DatasetBrowser.svelte` — masonry grid with lazy loading
7. Frontend: `ImageDetail.svelte` — focused view with caption display
8. Wire up pagination with next_token pattern

### Phase 3: Captioning Integration

1. ~~`yadc/api/modules/event_dispatcher.py`~~ ✅ — Done (subscribe/dispatch, @event_handler decorator)
2. `yadc/api/controllers/api_captioning.py` — start/stop captioning in background thread, SSE progress
3. `CaptionSettings.svelte` — config form for captioning options
4. Wire SSE events for real-time progress (images done, tokens, errors)
5. Caption display updates as images are processed

### Phase 4: Config & Export Management

1. `yadc/api/controllers/api_configs.py` — CRUD for environments, templates, user configs
2. `yadc/api/controllers/api_export.py` — trigger exports
3. `SettingsDialog.svelte` — full settings UI
4. Export form with backend/format selection

---

## Backend API Endpoints (planned)

| Method | Path | Description | Phase |
|--------|------|-------------|-------|
| GET | `/api/datasets` | List available datasets | 2 |
| GET | `/api/datasets/{name}/images` | List images with captions/drafts (paginated) | 2 |
| GET | `/api/datasets/{name}/images/{id}/media` | Serve image file | 2 |
| GET | `/api/datasets/{name}/images/{id}/thumbnail` | Serve/generated thumbnail | 2 |
| GET | `/api/datasets/{name}/images/{id}/caption` | Get caption text + TOML extras | 2 |
| PUT | `/api/datasets/{name}/images/{id}/caption` | Update caption/extras | 2 |
| POST | `/api/datasets/{name}/caption` | Start captioning run | 3 |
| GET | `/api/datasets/{name}/caption/status` | SSE stream for captioning progress | 3 |
| DELETE | `/api/datasets/{name}/caption` | Stop captioning run | 3 |
| GET | `/api/configs` | List user configs | 4 |
| GET | `/api/envs` | List environments | 4 |
| GET | `/api/templates` | List prompt templates | 4 |
| GET | `/api/templates/{name}` | Get template content | 4 |
| PUT | `/api/templates/{name}` | Update template | 4 |
| POST | `/api/export` | Trigger export | 4 |
| GET | `/api/events` | Global SSE event stream | 3 |

---

## Key Differences from Reference

| Aspect | qwen-reranker-test | yadc |
|--------|-------------------|------|
| Domain | Gallery search with ML embeddings | Dataset captioning tool |
| Core operations | Search, browse media | Caption generation, editing, export |
| Real-time needs | Ingestion status, new documents | Captioning progress per image |
| Data model | Items in SQLite | Images on disk with .txt/.toml sidecars |
| Heavy computation | Model inference on search | API-based captioning (no local model) |
| Config | In-code dataclass defaults | TOML files + user configs + envs |
| DI usage | Full injector from the start | Full injector + auto-discovery (`discover_services`/`discover_controllers`) |
| Env vars | `$env/dynamic/public` | `$lib/api.ts` constant (simpler for static builds) |
