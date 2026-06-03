---
name: todo
description: Deferred tasks, known issues, and future improvements. Check before starting new work to avoid duplicating known problems.
category: meta
priority: 3
---

# TODO

## ~~API request consolidation~~ **DONE**

All 4 issues resolved. See **plans/api-request-consolidation-plan** for full details.

- ✅ (a) `GET /api/envs` returns full `EnvInfo[]` details, not just names
- ✅ (b) Dataset page requests consolidated via `debounce()` dedupe
- ✅ (c) Env details loaded once via enriched list endpoint + SSE auto-refresh
- ✅ (d) `ImageCaptionedEvent` includes `caption: str` — SSE caption prefetch

## ~~Convert API backend to async (ASGI)~~ **DONE** — migrated to Quart + uvicorn

The API backend was migrated from Flask/waitress to Quart/uvicorn in two phases:
- **Phase 1**: Framework swap (Flask → Quart, waitress → uvicorn, `async def` on `send_file` routes).
- **Phase 2**: Fully async SSE (`asyncio.Queue` per client, `EventDispatcher` async bridge, native `async for` in `/events`).
- **Phase 4**: Graceful shutdown timeout (`timeout_graceful_shutdown=5`), integration tests for SSE connection cleanup, event delivery, and EventDispatcher thread-to-async bridging.

All success criteria met: SSE connects/reconnects/resumes, Ctrl+C shuts down cleanly within 2s (hard timeout at 5s), no thread blocking, `pytest` passes.

## Test cleanup

Several test files need structural cleanup:

- **Awkward structure**: `test_dataset_resolver.py` and `test_cli_draft.py` use class-based test organization that doesn't match the rest of the codebase (top-level functions). Should be refactored to use flat `def test_*` functions.
- **Wrong location**: `test_cli_draft.py` lives at `tests/test_cli_draft.py` (root of `tests/`) instead of `tests/cli/test_cli_draft.py` alongside the other CLI tests.
- **Inconsistent patterns**: Some tests use classes (`class TestXxx`), some use modules with top-level functions. Standardize on top-level `def test_*` functions.
- **Shared fixtures**: Consider consolidating the `cli` fixture usage patterns — `isolated=True` vs `env="..."` — into a clearer naming convention (e.g. `cli_isolated` / `cli_integration` fixtures).

### Deferred: FastAPI migration
A future FastAPI migration is possible but intentionally deferred. Quart is API-compatible with Flask and validated. FastAPI would give native Pydantic request/response models (fixing 21 basedpyright warnings in controllers) and automatic OpenAPI docs, but requires rewriting every route to use `Depends()` instead of injector closures. Not worth the churn until Quart proves problematic.

Key files changed: `yadc/api/application.py`, `yadc/api/controllers/`, `yadc/api/modules/sse_events.py`, `yadc/api/modules/event_dispatcher.py`, `pyproject.toml`.

## Stray node_modules .py file in wheel builds

When building wheels with `yadc/webui/__init__.py` present (needed for `package-data` to include `build/**/*`), setuptools discovers `yadc/webui/node_modules/flatted/python/flatted.py` as a submodule and includes it in the wheel. This is a harmless 3.7KB file but shouldn't be there. `exclude-package-data` doesn't work because setuptools treats it as a module, not data. Proper fix: use explicit `packages = [...]` list in pyproject.toml instead of `packages.find`, or filter out node_modules at the sdist level.

## Sidebar / topbar refactor

The navbar was moved from a horizontal top bar to a left sidebar (icon-rail on desktop, slide-in overlay on mobile). A topbar pattern was introduced for page-level header content (title, status). This is functional but needs cleanup:

- **UI refinement**: The sidebar and topbar need a visual polish pass — spacing, sizing, visual consistency
- **Topbar padding / height inconsistency**: The dataset listing (`#/`) and templates (`#/templates`) pages feel cramped below the topbar because `.app-topbar` has `padding-bottom: 0`. Adding bottom padding globally causes the dataset browser (`#/datasets/:name`) to shift down because its subtitle row uses `min-h-7` to reserve space for the captioning status row, which is taller than the idle text line. The hamburger button and subtitle also shift slightly when captioning starts/stops. **Superseded by `captioning-status-bar-plan`** — moving the progress UI out of the topbar entirely.

## Frontend remaining cleanup

- SidePanel should accept a `class` prop — its positioning (inline vs fixed, width, etc.) is controlled by the parent page, not internal to the component
- **Tooltip z-index / stacking context**: `Tooltip.svelte` was migrated to Tailwind but the tooltip label renders underneath `DatasetImage` tiles. The `DatasetBrowser` tiles use `transform` (hover scale) and `relative` positioning, which create new stacking contexts. The old tooltip had `z-index: 50` but that alone is not sufficient when the parent stacking contexts are lower. The tooltip should not hardcode its own z-index; it should be parent-driven (e.g. via a `class` prop on the wrapper `div` or a dedicated `zIndex` prop). Deciding the right API is deferred. Key files: `Tooltip.svelte`, `DatasetBrowser.svelte`, `DatasetImage.svelte`.

## Split `stores/datasetImages.ts` (656 lines) — data-layer refactor

Deferred from `plans/frontend-component-organization.md`. This is a single file holding: types (~60 lines), every API helper for datasets/images/captions/uploads (~400 lines), and the `createDatasetBrowserStore` factory (~100 lines). The "too big" problem is the same as the component-organization work but it's a data-layer concern. Proposed split:

- `stores/datasetImages/types.ts` — `DatasetInfo`, `ImageInfo`, `ImagePage`, `CaptionData`, `HistoryEntry`, `DatasetUploadResult`, `UploadConflict`, `UploadProgressEvent`, `CaptioningJobInfo`, `DatasetBrowserState`, `DatasetFolder`
- `stores/datasetImages/api.ts` — all the fetch/upload/update helpers (debounced + raw)
- `stores/datasetImages/browser-store.ts` — `createDatasetBrowserStore` factory + URL helpers (`mediaUrl`, `thumbnailUrl`)
- `stores/datasetImages/index.ts` — re-exports for back-compat (everything currently imported from `$lib/stores/datasetImages` continues to work)

Risk: `+page.svelte` and several other files import the entire namespace from one barrel — if the import path changes mid-refactor, lots of `import type` lines need to update. Use the index.ts re-export strategy to keep the import surface stable.

## ~~Missing webui assets~~ **DONE** — Logo added (android-chrome-192x192.png used in sidebar)

## Webui code quality pass

The webui frontend code is newly written and needs a cleanup pass to bring it up to a higher standard:

- **Error handling**: Consistent error boundaries, user-facing error messages (toast/banner instead of silent failures), proper handling of API error responses in stores
- **Models/types**: Review and tighten TypeScript types — ensure API response shapes are well-typed, avoid `any`, consider Zod schemas for API response validation (already used for SSE events)
- **Code style**: Consistent patterns for component structure, prop definitions, store usage, `$effect` cleanup, etc. Reduce duplication across similar components (e.g. the various dialog components)
- **General cleanup**: Remove dead code, unused imports, consolidate shared logic
- **Basedpyright warnings**: Fix all type-checking warnings in the webui-related backend code (and elsewhere)
- **API controller basedpyright warnings**: `api_configs.py`, `api_datasets.py`, and `api_envs.py` have 21 pre-existing warnings (as of the error-response refactor). Root cause is Quart's untyped `request.get_json()` / `resp.json()` returning `Any`, so pyright flags every downstream access as `Unknown`. Fixing these requires either: (1) adding request/response Pydantic models and validating at the boundary, (2) using `typing.cast()` or `assert isinstance(...)` guards that pyright understands, or (3) adding `# type: ignore` comments with explanatory notes where runtime checks already guarantee safety.
- **Watchdog Observer type**: `Observer` from `watchdog` is currently typed as `Any` to suppress basedpyright errors — this needs a proper fix. Investigate why basedpyright can't resolve `watchdog.observers.Observer` (likely missing/incomplete stubs) and find the right solution (e.g. custom stub, `type: ignore` with comment, or wrap with a protocol)

## UI/UX overhaul

The current UI is functional but needs a manual pass to improve overall look and feel. This is a larger undertaking covering:

- Visual polish: spacing, typography, color consistency, border/shadow usage, hover/focus/active states
- UX improvements: better loading states, empty states, error feedback, confirmation prompts for destructive actions
- Responsive layout — ensure the UI works well at different viewport sizes
- Accessibility basics — keyboard navigation, ARIA attributes, focus management in dialogs
- Overall design coherence — the UI should feel like a unified application rather than assembled parts

## Simplified dataset config editing

See **plans/dataset-config-settings-plan** for full details. Mostly done:

- [x] `PATCH /configs/<name>` endpoint (JSON body, deep-merged into TOML)
- [x] `rounds` wired through backend + frontend
- [x] Frontend fetches config defaults + pre-fills fields
- [x] Diff indicators (dots) + collapsible overrides section with reset
- [x] Removed `ConfigEditor.svelte` from Settings dialog
- [x] Phase 4: type-safe Config API (Pydantic validation on GET/PATCH, typed TS interfaces)
- [ ] TOML multiline string serialization for templates
- [x] "Save as dataset default" action (Config tab in side panel, PATCH via `DatasetConfig.svelte`)

## Test captioning flow in the webui

The full captioning workflow (start → progress → completion → result display) needs end-to-end testing through the webui to catch any integration issues between the frontend stores, SSE events, and the backend captioning API.

## Dataset browser scroll cutoff

The dataset browser grid uses `overflow-y-auto` on its parent div, but images near the bottom get cut off because the scroll container's padding doesn't extend past the last items. The fix is to replace the padding-based spacing on the scroll container with margin-based spacing on the children (grid items), so the last row of images is fully visible when scrolled to the bottom.

## Watchdog job_id tag cleanup strategy

~~The `DatasetWatcherService._expected_sources` dict maps `dataset_name → job_id` so that `DatasetChangedEvent`s carry the `job_id` of the captioning operation that caused them. This lets the initiating frontend suppress the "Dataset files have changed" banner.~~ **DONE** — Replaced by the three-tier expected-change tracking system in `DatasetWatcherService` (see `docs/dataset-watcher.md`):

- **Dataset-level** (`_expected_sources`): tags the dataset-level `job_id` for captioning jobs; cleared 5s after the job ends.
- **File-level** (`_expected_files`): per-file TTL-based tracking via `expect_file_change(name, path, *, source)`. `source` is `"ui:<clientId>"` for webui edits (per-tab UUID) or `SELF_JOB_ID = "self"` for backend-only callers. Used by `DatasetService` (caption saves, extras updates, history restores, image deletes) and `AsyncCaptionJob._acaption_one()` (captioning writes).
- **Pattern-level** (`_expected_patterns`): `fnmatch` globs for bulk deletions (`folder/*`, `STEM.*.draft~`); same TTL/deque semantics. Used by `ManagedDatasetsService.delete_items` (folder deletions) and the draft-delete path.

All three are bounded deques (`watcher_expected_file_max`, default 256) with TTL expiry (`watcher_expected_file_ttl`, default 1.0s). Entries are NOT consumed on match because a single write can produce multiple inotify events. The frontend suppresses the event when the dispatched `job_id` matches its own `clientId`, an active captioning job_id, or the legacy `"self"` sentinel. The original race-condition concern is moot because file-level matching is the primary suppression mechanism — the dataset-level job_id is just a tag for downstream filtering.

## SSE captioning event reliability / resumption

**DONE** — Implemented `Last-Event-ID` based SSE event resumption:

- **Backend**: `SSEEvents` maintains a configurable ring buffer (`sse_event_history_size`, default 128) of recent non-ping events with monotonic IDs. On reconnect, the controller reads `Last-Event-ID` from the request header and replays missed events from history. If the requested ID is too old (no longer in the buffer), a `ResumptionFailedEvent` is sent to the client.
- **Backend**: `api_events.py` sends `id:` fields on every non-ping SSE event and `retry: 5000` at connection start so the browser auto-reconnects after 5s.
- **Frontend**: Switched from manual close/reconnect to browser's built-in `EventSource` auto-reconnect, which automatically preserves and sends `Last-Event-ID`. Added `resumptionFailed` store and warning banner on the dataset detail page with Refresh/Dismiss actions.
- Key files: `yadc/api/modules/sse_events.py`, `yadc/api/controllers/api_events.py`, `yadc/api/events.py` (`ResumptionFailedEvent`), `yadc/api/configuration.py` (`sse_event_history_size`), `yadc/webui/src/lib/stores/events.ts`, `yadc/webui/src/routes/datasets/[name]/+page.svelte`.

### Remaining edge cases

- ~~**Mid-captioning page load**: If a user opens the dataset page while captioning is already running, they won't see any progress until the next SSE event arrives.~~ **DONE** — Added `GET /datasets/<name>/caption` endpoint and a one-time fetch in the dataset page's `$effect` (guarded to only seed when the store is idle, so it never clobbers live SSE events).
- ~~**WSGI thread exhaustion**: Each SSE connection still blocks a thread. The async migration (see top-level TODO) is the proper fix.~~ **DONE** — SSE is fully async since the Quart migration.

## Caption settings: dataset defaults integration

See **plans/dataset-config-settings-plan** for full details. Phase 1 + 2 mostly done.

Remaining:
- "Save as dataset default" action — writes current settings back to TOML via `PATCH /configs/<name>`
- Preset profiles, config diff banner (Phase 3 nice-to-haves)
- ~~Phase 4: type-safe config API (Pydantic validation on GET/PATCH)~~ **DONE** — see `plans/dataset-config-settings-plan`

## History restore UI and semantics

The first pass of history browsing/restoring is implemented (backend API + frontend UI). Needs refinement:

- **UI polish**: The history section in ImageDetail needs visual improvement — better layout, spacing, differentiation between entries
- **Current state in history**: History always includes the current state as the most recent entry, which means restoring always adds a duplicate (current state gets saved again). Need to decide: should history only contain *past* states? Should the frontend filter out the current state? Should restore skip saving if the current state is already the target?

## Incremental filesystem index updates

~~Currently, `DatasetChangedEvent` triggers a full `rescan_dataset()` which re-scans every file in every directory of the dataset. This is wasteful when only one file was added or removed.~~ **PARTIAL** — Diff scans are in: `DatasetService._apply_disk_scan` now compares the freshly-walked disk state against the current index (via `DatasetRepository.list_image_infos`) and only writes SQL for changed rows (`to_upsert` + `to_delete`). The function returns `bool` so callers (`_refresh_stale_datasets` background job and `rescan_dataset` API endpoint) only dispatch `DatasetChangedEvent` when the scan found real changes. The `_refresh_stale_datasets` interval is now `dataset_refresh_interval_seconds` (default 300s) and runs in a background thread via `JobScheduler` (with `_refresh_lock` to serialize against manual rescans).

**Still pending**:
- Pass the affected file path(s) in the event (or a new granular event type like `DatasetFileAddedEvent` / `DatasetFileRemovedEvent`) so `DatasetService` can do targeted SQLite upserts/deletes for the affected files **without** walking the whole dataset. Currently every external change still triggers a full disk walk.

## Unify DatasetImage resolution for webui preview and captioning

`DatasetService.preview_prompt()` manually constructs `DatasetImage` instances (reading caption, TOML extras, drafts) with ad-hoc code that diverges from `read_image_from_disk()` used by `resolve_dataset()` in the captioning pipeline. This duplication caused the caption to be missing from the template preview context (fixed with a one-liner). A single shared resolution function (e.g. `DatasetImage.from_path()` or a service-level helper) should be used by both paths to prevent similar regressions. Key files: `yadc/api/services/datasets.py` (`preview_prompt`), `yadc/core/dataset_resolver.py` (`read_image_from_disk`).

## Clean up captioning server logs

The API captioning service (`CaptioningService` / `AsyncCaptionJob`) reuses CLI-level code (`APICaptioner`, `cmd_envs`, `cmd_templates`, `resolve_dataset`, etc.) which logs verbosely to stdout/stderr using print statements and CLI-style formatters (progress bars, usage stats, interactive prompts). When captioning via the API/webui, these logs pollute the server output. The logging needs a pass to:
- Replace print/prompt output with proper `logger` calls at appropriate levels
- Ensure `APICaptioner` and shared `cmd/` modules use structured logging instead of direct stdout
- Suppress or quiet CLI-specific output (progress bars, interactive menus) when running in API mode
- Review `AsyncCaptionJob._ado_run()` and its callees for noisy output


## TOML config revision history

`PATCH /configs/<name>` writes directly to the TOML file with no backup or history. Users can accidentally break their config and have no way to revert. We should track historical versions so the WebUI can offer "revert to previous version".

**Scope**: imported (`source="import"`) and created (`source="create"`) datasets reference a single external TOML file. Uploads (`source="upload"`) have a TOML generated by yadc in the state dir.

**Possible approaches**:
- Keep a `.history~` file next to the TOML (same pattern as image caption history). On every `PATCH`, append the previous content as a timestamped entry. The WebUI can list entries and restore by overwriting the TOML.
- Use a lightweight VCS (e.g. `git init` in the dataset state dir, commit on each change). Overkill but gives full diffs.
- Store revisions in SQLite (new `config_history` table). Simpler but loses the "file is the source of truth" property.

**Key files**: `yadc/api/services/configs.py` (`patch_config`), `yadc/api/controllers/api_configs.py`, `yadc/webui/src/lib/stores/configs.ts`, `yadc/webui/src/routes/datasets/[name]/DatasetConfig.svelte`.

## TOML comment preservation on API write-back

`PATCH /configs/<name>` and `PUT /configs/<name>` parse the TOML, merge changes, and re-serialize. Any comments in the original file are lost because the TOML data model has no concept of comments — all TOML libraries discard them on parse. This is fine for configs created/managed through the webui, but imported configs that the user authored with comments will have them stripped on the first edit. Possible approaches:
- Text-level patching (find/replace in the raw string instead of parse→serialize) — works for simple scalar changes but can't handle structural changes
- A TOML AST-aware library that preserves comments and formatting (e.g. `taplo`/Python bindings if they exist)
- Accept the limitation and document it (comments are not preserved when editing configs through the webui)

## Normalize draft_names storage in SQLite

The `draft_names` column in `dataset_images` stores draft names as a comma-separated string (e.g. `"gemma,qwen"`). This is fragile — LIKE-based queries need 4 OR clauses to match a single draft name (`search_by_draft_name`), and counting/splitting happens in Python, not SQL. Should be normalized to a proper join table (`draft_names` → `image_drafts` table with `image_id` + `draft_name` columns).

This was highlighted when adding `get_draft_name_counts()` and `search_by_draft_name()` to `DatasetRepository` — both work around the CSV format with Python-side splitting or 4-way LIKE matching.

Key files: `yadc/api/services/dataset_repository.py` (schema + queries), `yadc/api/modules/db_migrations.py` (migration), `yadc/api/services/datasets.py` (service methods that read/write `draft_names`).

## SSE event pattern standardization

Research whether to standardize on thin events (notify-then-fetch) vs event-carried state transfer (fat events) for SSE. See `todo/thin-events-vs-fat-events.md` for full context.

**Partial decision made**: `EnvironmentsChangedEvent` and `TemplatesChangedEvent` carry the full list of changed names (`envs: list[str]`, `templates: list[str]`). The frontend still calls `refreshEnvs()`/`refreshTemplates()` on these events (the lists are for debugging/future use). The pattern is "enriched thin events" — notify with context, then fetch for authoritative state.

# User TODOs (less verbose)

* errors when starting captions show up in both the toast and at the top (latter needs removal)
* error toasts have no details (just says HTTP 502)
* need to enable prettier
* ~~need to fix the navbar at the top on mobile (it cuts off, export/settings not visible when selecting a dataset)~~
* ~~on mobile, cannot edit datasets or templates (hover-only action buttons)~~
* **Decide on `_metadata` field for env GET endpoints**: Consider grouping read-only metadata (`has_token`, `token_method`) under a `_metadata` key to make the PUT/GET shape symmetry explicit. Currently kept flat for simplicity, but worth revisiting if more read-only fields are added later.
* **Firefox drag-and-drop broken on dataset browser**: File drops on the dataset browser page (`#/datasets/:name`) don't trigger in Firefox — the overlay never appears. Works fine in Chromium and works in the Add Files dialog (FileDropZone) in both browsers. A speculative `DOMStringList` fix was stashed but didn't resolve it. Needs real investigation. Stash: `wip: Firefox DOMStringList fix for dataset browser drag-and-drop`.
* **SSE `/api/events` disconnects every ~60s in Firefox**: The browser closes the connection and auto-reconnects (no uvicorn logs, not a keep-alive timeout). Added `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers (best practice for SSE, though they didn't fix this specific issue). Functionally harmless because of `Last-Event-ID` resumption, but root cause unknown. Tracking only — not actively investigating.
