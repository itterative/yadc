---
name: todo
description: When a user query involves deferred tasks, known issues, or future improvements, read this memory first.
---

# TODO

## ~~Convert API backend to async (ASGI)~~ **DONE** — migrated to Quart + uvicorn

The API backend was migrated from Flask/waitress to Quart/uvicorn in two phases:
- **Phase 1**: Framework swap (Flask → Quart, waitress → uvicorn, `async def` on `send_file` routes).
- **Phase 2**: Fully async SSE (`asyncio.Queue` per client, `EventDispatcher` async bridge, native `async for` in `/events`).
- **Phase 4**: Graceful shutdown timeout (`timeout_graceful_shutdown=5`), integration tests for SSE connection cleanup, event delivery, and EventDispatcher thread-to-async bridging.

All success criteria met: SSE connects/reconnects/resumes, Ctrl+C shuts down cleanly within 2s (hard timeout at 5s), no thread blocking, `pytest` passes.

### Deferred: FastAPI migration
A future FastAPI migration is possible but intentionally deferred. Quart is API-compatible with Flask and validated. FastAPI would give native Pydantic request/response models (fixing 21 basedpyright warnings in controllers) and automatic OpenAPI docs, but requires rewriting every route to use `Depends()` instead of injector closures. Not worth the churn until Quart proves problematic.

Key files changed: `yadc/api/application.py`, `yadc/api/controllers/`, `yadc/api/modules/sse_events.py`, `yadc/api/modules/event_dispatcher.py`, `pyproject.toml`.

## Stray node_modules .py file in wheel builds

When building wheels with `yadc/webui/__init__.py` present (needed for `package-data` to include `build/**/*`), setuptools discovers `yadc/webui/node_modules/flatted/python/flatted.py` as a submodule and includes it in the wheel. This is a harmless 3.7KB file but shouldn't be there. `exclude-package-data` doesn't work because setuptools treats it as a module, not data. Proper fix: use explicit `packages = [...]` list in pyproject.toml instead of `packages.find`, or filter out node_modules at the sdist level.

## Frontend remaining cleanup

- SidePanel should accept a `class` prop — its positioning (inline vs fixed, width, etc.) is controlled by the parent page, not internal to the component

## Missing webui assets

- **Logo** — no app logo exists for use in the navbar/header and potentially as a larger brand mark. Theme-matching to be done later.

## Webui code quality pass

The webui frontend code is newly written and needs a cleanup pass to bring it up to a higher standard:

- **Error handling**: Consistent error boundaries, user-facing error messages (toast/banner instead of silent failures), proper handling of API error responses in stores
- **Models/types**: Review and tighten TypeScript types — ensure API response shapes are well-typed, avoid `any`, consider Zod schemas for API response validation (already used for SSE events)
- **Code style**: Consistent patterns for component structure, prop definitions, store usage, `$effect` cleanup, etc. Reduce duplication across similar components (e.g. the various dialog components)
- **General cleanup**: Remove dead code, unused imports, consolidate shared logic
- **Basedpyright warnings**: Fix all type-checking warnings in the webui-related backend code (and elsewhere)
- **API controller basedpyright warnings**: `api_configs.py`, `api_datasets.py`, and `api_envs.py` have 21 pre-existing warnings (as of the error-response refactor). Root cause is Flask's untyped `request.get_json()` / `resp.json()` returning `Any`, so pyright flags every downstream access as `Unknown`. Fixing these requires either: (1) adding request/response Pydantic models and validating at the boundary, (2) using `typing.cast()` or `assert isinstance(...)` guards that pyright understands, or (3) adding `# type: ignore` comments with explanatory notes where runtime checks already guarantee safety.
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

The `DatasetWatcherService._expected_sources` dict maps `dataset_name → job_id` so that `DatasetChangedEvent`s carry the `job_id` of the captioning operation that caused them. This lets the initiating frontend suppress the "Dataset files have changed" banner.

**Problem**: The tag is never cleared, so any future filesystem event for that dataset (e.g. adding a new image hours later) would carry the stale `job_id`. If that `job_id` is still in the frontend's 16-item list, the notification would be incorrectly suppressed.

**Challenge**: There's a fundamental race condition — we can't know when the *last* inotify event from a captioning session has been dispatched. Any timer-based clear is a heuristic:
- Too short: late inotify events arrive after the tag is cleared → untagged event → spurious notification on the initiating frontend
- Too long: legitimate external changes get tagged with a stale `job_id` → suppressed incorrectly

**Possible approaches**:
1. Timer-based clear with generous delay (e.g. 3-5x debounce interval). Pragmatic but racy.
2. Have `CaptioningService` listen for `DatasetChangedEvent` via `@event_handler` and re-tag them directly — avoids the watcher tag entirely but is architecturally complex.
3. Track a monotonic "captioning generation" per dataset in the watcher — `_dispatch_change` consumes the tag but stores it for a short grace period so the next event can still match.
4. Use a different mechanism entirely (e.g. per-request SSE filtering instead of event tagging).

The current implementation (never clearing) is being reverted. The job_id infrastructure in `DatasetChangedEvent`, `CaptioningService`, and the frontend `registerJobId`/bounded list should remain — only the watcher tag lifecycle needs a proper solution.

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

## Incremental filesystem index updates

Currently, `DatasetChangedEvent` triggers a full `rescan_dataset()` which re-scans every file in every directory of the dataset. This is wasteful when only one file was added or removed.

**Fix**:
- Make `DatasetWatcherService` pass the affected file path(s) in the event (or a new granular event type like `DatasetFileAddedEvent` / `DatasetFileRemovedEvent`).
- `DatasetService` should handle these by doing targeted SQLite upserts/deletes for the affected files instead of a full rescan.
- Increase or remove the `_refresh_stale_datasets` interval (currently 60s) since the watcher now handles updates incrementally — the stale refresh becomes just a fallback.

## Clean up captioning server logs

The API captioning service (`CaptioningService` / `CaptionJob`) reuses CLI-level code (`APICaptioner`, `cmd_envs`, `cmd_templates`, `resolve_dataset`, etc.) which logs verbosely to stdout/stderr using print statements and CLI-style formatters (progress bars, usage stats, interactive prompts). When captioning via the API/webui, these logs pollute the server output. The logging needs a pass to:
- Replace print/prompt output with proper `logger` calls at appropriate levels
- Ensure `APICaptioner` and shared `cmd/` modules use structured logging instead of direct stdout
- Suppress or quiet CLI-specific output (progress bars, interactive menus) when running in API mode
- Review `CaptionJob._do_run()` and its callees for noisy output


## TOML comment preservation on API write-back

`PATCH /configs/<name>` and `PUT /configs/<name>` parse the TOML, merge changes, and re-serialize. Any comments in the original file are lost because the TOML data model has no concept of comments — all TOML libraries discard them on parse. This is fine for configs created/managed through the webui, but imported configs that the user authored with comments will have them stripped on the first edit. Possible approaches:
- Text-level patching (find/replace in the raw string instead of parse→serialize) — works for simple scalar changes but can't handle structural changes
- A TOML AST-aware library that preserves comments and formatting (e.g. `taplo`/Python bindings if they exist)
- Accept the limitation and document it (comments are not preserved when editing configs through the webui)

# User TODOs (less verbose)

* errors when starting captions show up in both the toast and at the top (latter needs removal)
* error toasts have no details (just says HTTP 502)
* need to enable prettier
* ~~need to fix the navbar at the top on mobile (it cuts off, export/settings not visible when selecting a dataset)~~
* ~~on mobile, cannot edit datasets or templates (hover-only action buttons)~~
