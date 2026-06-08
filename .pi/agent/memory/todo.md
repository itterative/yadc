---
name: todo
description: Deferred tasks, known issues, and future improvements. Check before starting new work to avoid duplicating known problems.
category: meta
priority: 3
---

# TODO

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

## Sidebar / topbar refactor (visual polish)

The navbar was moved from a horizontal top bar to a left sidebar (icon-rail on desktop, slide-in overlay on mobile). A topbar pattern was introduced for page-level header content (title, status). The structural move is done; this entry is now about **visual polish** (not part of `plans/archive/frontend-component-organization.md` which is purely structural).

- **UI refinement**: The sidebar and topbar need a visual polish pass — spacing, sizing, visual consistency

## Standardize password-passing on a single mechanism (header)

The current API has **inconsistent** ways for the client to pass a decryption password to endpoints that decrypt env settings:

- `POST /envs/<name>/reveal` — `{"password": "..."}` in the request body
- `PUT /envs/key-mode` — `{"password": "..."}` in the request body
- `POST /envs/<name>/models` (added 2026-06-05, see `plans/list-models-captioner-reuse-plan`) — `{"password": "..."}` in the request body

The original `list_models` was `GET` (no body) and the controller could only surface a 403 `PASSWORD_REQUIRED` — the client had no way to actually supply a password. We extended it to `POST` with a body, matching the `reveal_env_value` / `set_key_mode` precedent.

This is fine for now but **header-based credential passing is the better long-term shape**:

- Conventional (mirrors `Authorization: Bearer ...`, `X-API-Key`, etc.)
- Not logged in URLs / browser history / request bodies
- Avoids the awkward `@app.route(..., methods=["GET", "POST"])` dual-handler pattern that exists only because GETs can't carry a body
- Standardizes the `YADC_PASSWORD` env-var fallback + custom-header override pattern

**Migration plan** (deferred):

1. Pick a header name — `X-YADC-Password` is the natural choice (sibling of `Authorization`).
2. Add a small helper on the controller side: `_resolve_request_password(headers) -> str | None` that reads the header and falls back to `YADC_PASSWORD` (mirrors what `cmd_envs.decrypt_setting` already does internally).
3. Convert each of the three POST-with-body endpoints to GET with a `X-YADC-Password` header. Keep backward compatibility by reading the body too (warn-once if both are set) for one release.
4. Drop the body-reading code in each handler.
5. Update the frontend `revealEnvValue` / `setKeyMode` / `fetchModels` to use the header.

Tracking under TODO (not actively scheduled) — the current POST-with-body approach works and matches existing patterns, so there's no urgent need to migrate.

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

See **plans/dataset-config-settings-plan** for full details. Mostly done — remaining:

- [ ] TOML multiline string serialization for templates

## Test captioning flow in the webui

The full captioning workflow (start → progress → completion → result display) needs end-to-end testing through the webui to catch any integration issues between the frontend stores, SSE events, and the backend captioning API.

## Dataset browser scroll cutoff

The dataset browser grid uses `overflow-y-auto` on its parent div, but images near the bottom get cut off because the scroll container's padding doesn't extend past the last items. The fix is to replace the padding-based spacing on the scroll container with margin-based spacing on the children (grid items), so the last row of images is fully visible when scrolled to the bottom.

## Caption settings: dataset defaults integration

See **plans/dataset-config-settings-plan** for full details. Phase 1 + 2 mostly done.

Remaining (Phase 3 nice-to-haves):
- Preset profiles
- Config diff banner

## History restore UI and semantics

The first pass of history browsing/restoring is implemented (backend API + frontend UI). Needs refinement:

- **UI polish**: The history section in ImageDetail needs visual improvement — better layout, spacing, differentiation between entries
- **Current state in history**: History always includes the current state as the most recent entry, which means restoring always adds a duplicate (current state gets saved again). Need to decide: should history only contain *past* states? Should the frontend filter out the current state? Should restore skip saving if the current state is already the target?

## Incremental filesystem index updates

**DONE** — Watcher-level change detection now plumbs affected paths through to the dataset service for targeted index updates.

- `DatasetChangedEvent` carries a new optional `changed_paths: list[str]` field (empty for legacy/manual callers).
- `DatasetWatcherService` accumulates the per-dataset set of changed paths in `_changed_paths` during the debounce window and ships it on the dispatched event. `on_moved` records both `src_path` and `dest_path` so renames don't drop the new file.
- `DatasetService._on_dataset_changed` dispatches to `DatasetScanner.scan_targeted` when the event has `changed_paths` populated, falling back to `DatasetScanner.scan_disk` (full walk) for legacy/empty cases. Self-originated events (`job_id` set) are still skipped — the API endpoints keep the index in sync.
- `DatasetScanner.scan_targeted` maps each changed path to its image row(s) via `resolve_affected_image_paths` (strips `.txt`/`.toml`/`.history~` to get the image stem; expands drafts to `<stem>.<ext>` candidates and filters to rows in the index), stats each candidate, and does targeted `upsert_image` / `delete_image` for the affected rows only. `update_dataset_stats` is updated with the new count (computed locally: `len(existing) + new_rows - len(to_delete)`).
- **Refactor (post-completion)**: disk-scanning code was extracted out of `DatasetService` into `DatasetScanner` (`scan_disk`, `scan_targeted`, `read_disk`, `resolve_affected_image_paths`, `scan_image_meta`, `_image_meta_differs`, `IMAGE_EXTENSIONS`). Config-loading helpers (`load_config`, `load_raw_config`, `resolve_relative_paths`, `get_dataset_paths`) went to a new `DatasetLoader` service. `DatasetService` shrank from 1300+ to 891 lines and now orchestrates both. Tests split into `test_dataset_scanner.py` and `test_dataset_loader.py`; `test_datasets_service.py` kept the lifecycle/CRUD coverage.
- `_scan_disk` and the new `_scan_image_meta` helper share the per-image metadata extraction logic.
- Tests added: 11 in `test_datasets_service.py` (resolver + targeted update) and 7 in `test_dataset_watcher.py` (changed-paths tracking).

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
* **Decouple ETA from captioningStatus store**: The ETA estimation is coupled to `captioningStatus` carrying `api_url`/`api_model_name`. A separate job-identity store should track `{api_url, api_model_name, job_id}` so `captioningStatus` only carries progress. See `todo/eta-decouple.md` for details.
* **Decide on `_metadata` field for env GET endpoints**: Consider grouping read-only metadata (`has_token`, `token_method`) under a `_metadata` key to make the PUT/GET shape symmetry explicit. Currently kept flat for simplicity, but worth revisiting if more read-only fields are added later.
* **Firefox drag-and-drop broken on dataset browser**: File drops on the dataset browser page (`#/datasets/:name`) don't trigger in Firefox — the overlay never appears. Works fine in Chromium and works in the Add Files dialog (FileDropZone) in both browsers. A speculative `DOMStringList` fix was stashed but didn't resolve it. Needs real investigation. Stash: `wip: Firefox DOMStringList fix for dataset browser drag-and-drop`.
* **SSE `/api/events` disconnects every ~60s in Firefox**: The browser closes the connection and auto-reconnects (no uvicorn logs, not a keep-alive timeout). Added `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers (best practice for SSE, though they didn't fix this specific issue). Functionally harmless because of `Last-Event-ID` resumption, but root cause unknown. Tracking only — not actively investigating.
