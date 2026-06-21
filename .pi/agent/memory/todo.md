---
name: todo
description: Deferred tasks, known issues, and future improvements. Check before starting new work to avoid duplicating known problems.
category: meta
priority: 3
---

# TODO

## IndexedDB for prompt-generator few-shot examples

The prompt generator (`/prompts`) currently persists the form
settings (env, api url/token/model, intent, focus) to localStorage
in `lib/stores/prompts/settings.svelte.ts`, but **drops the
few-shot examples** on page reload — their `image_data_url` payloads
are too large for localStorage's ~5–10MB quota and there's no
per-example provenance to re-fetch dataset images on load.

The right fix is IndexedDB: async, orders-of-magnitude more
capacity, and the natural place for arbitrary blobs. Migration
shape:
- Add a `source` field to `ExamplePair`:
  `{type: "manual"}` or `{type: "dataset", dataset: string, imageId: string}`.
- Persist everything (settings + examples) in IndexedDB; settings
  can stay in localStorage if size isn't a concern, but IDB is fine
  for both.
- On page load, re-fetch dataset-sourced examples via
  `GET /datasets/<name>/images/<id>/image`; manual uploads still
  need to be re-added (or also stored in IDB as a Blob).
- Update `ExamplesPanel` to show a placeholder chip for
  re-fetch-in-progress or unrecoverable examples.

Tracked in the prompt-generator plan history
(`history/prompt-generator-plan/003-prompt-form-settings-persistence.md`)
under "Deferred".

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

## Webui code quality pass

The webui frontend code is newly written and needs a cleanup pass to bring it up to a higher standard:

- **Error handling**: Consistent error boundaries, user-facing error messages (toast/banner instead of silent failures), proper handling of API error responses in stores
- **Models/types**: Review and tighten TypeScript types — ensure API response shapes are well-typed, avoid `any`, consider Zod schemas for API response validation (already used for SSE events)
- **Code style**: Consistent patterns for component structure, prop definitions, store usage, `$effect` cleanup, etc. Reduce duplication across similar components (e.g. the various dialog components)
- **General cleanup**: Remove dead code, unused imports, consolidate shared logic
- **Basedpyright warnings**: Fix all type-checking warnings in the webui-related backend code (and elsewhere)
- **API controller basedpyright warnings**: `api_configs.py`, `api_datasets.py`, and `api_envs.py` have 21 pre-existing warnings (as of the error-response refactor). Root cause is Quart's untyped `request.get_json()` / `resp.json()` returning `Any`, so pyright flags every downstream access as `Unknown`. Fixing these requires either: (1) adding request/response Pydantic models and validating at the boundary, (2) using `typing.cast()` or `assert isinstance(...)` guards that pyright understands, or (3) adding `# type: ignore` comments with explanatory notes where runtime checks already guarantee safety.
- **Watchdog Observer type**: `Observer` from `watchdog` is currently typed as `Any` to suppress basedpyright errors — this needs a proper fix. Investigate why basedpyright can't resolve `watchdog.observers.Observer` (likely missing/incomplete stubs) and find the right solution (e.g. custom stub, `type: ignore` with comment, or wrap with a protocol)

## Dataset browser scroll cutoff

The dataset browser grid uses `overflow-y-auto` on its parent div, but images near the bottom get cut off because the scroll container's padding doesn't extend past the last items. The fix is to replace the padding-based spacing on the scroll container with margin-based spacing on the children (grid items), so the last row of images is fully visible when scrolled to the bottom.

## Caption settings: dataset defaults integration

See **plans/dataset-config-settings-plan** for full details. Phase 1 + 2 mostly done.

Remaining (Phase 3 nice-to-haves):
- Preset profiles
- Config diff banner

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

## Normalize draft_names storage in SQLite

The `draft_names` column in `dataset_images` stores draft names as a comma-separated string (e.g. `"gemma,qwen"`). This is fragile — LIKE-based queries need 4 OR clauses to match a single draft name (`search_by_draft_name`), and counting/splitting happens in Python, not SQL. Should be normalized to a proper join table (`draft_names` → `image_drafts` table with `image_id` + `draft_name` columns).

This was highlighted when adding `get_draft_name_counts()` and `search_by_draft_name()` to `DatasetRepository` — both work around the CSV format with Python-side splitting or 4-way LIKE matching.

Key files: `yadc/api/services/dataset_repository.py` (schema + queries), `yadc/api/modules/db_migrations.py` (migration), `yadc/api/services/datasets.py` (service methods that read/write `draft_names`).

## SSE event pattern standardization

Research whether to standardize on thin events (notify-then-fetch) vs event-carried state transfer (fat events) for SSE. See `todo/thin-events-vs-fat-events.md` for full context.

**Partial decision made**: `EnvironmentsChangedEvent` and `TemplatesChangedEvent` carry the full list of changed names (`envs: list[str]`, `templates: list[str]`). The frontend still calls `refreshEnvs()`/`refreshTemplates()` on these events (the lists are for debugging/future use). The pattern is "enriched thin events" — notify with context, then fetch for authoritative state.

## CLI zip export support

The export system supports two output paths: filesystem (`run_export()` → `backend.run()`) and zip (`run_export_zip()` → `backend.run_zip()`). The Web UI uses both, but the CLI (`yadc/cli_export.py`) only calls `run_export()` — there is no `--zip` option and no zip-writing code path.

This means zip-only backends can't be used from the CLI. Concretely, `yadc export --backend yadc` (the raw sidecar-archive backend added in the export-system work) always raises, because its `run()` rejects filesystem output with a clear message pointing at `run_export_zip()`.

To close the gap:
- Add a `--zip` / `--output <path.zip>` mode to `yadc/cli_export.py` that calls `run_export_zip()` and writes the returned `BytesIO` to the given path.
- Detect zip-only backends (`descriptor.zip_only`) and force zip output (mirroring `api_export.py`), rather than letting the user hit the `run()` error.
- The backend dispatch (`run_export`/`run_export_zip`) already exists; this is CLI-only plumbing.

Key files: `yadc/cli_export.py`, `yadc/core/exporters/__init__.py` (`_BackendDescriptor.zip_only`, `run_export_zip`). See **export-system** doc.

# User TODOs (less verbose)

* errors when starting captions show up in both the toast and at the top (latter needs removal)
* error toasts have no details (just says HTTP 502)
* need to enable prettier
* **Decouple ETA from the captioningStatuses store**: The ETA estimation is coupled to per-dataset status entries carrying `api_url`/`api_model_name`. A separate job-identity store should track `{api_url, api_model_name, job_id}` so the status entries only carry progress. See `todo/eta-decouple.md` for details.
* **Decide on `_metadata` field for env GET endpoints**: Consider grouping read-only metadata (`has_token`, `token_method`) under a `_metadata` key to make the PUT/GET shape symmetry explicit. Currently kept flat for simplicity, but worth revisiting if more read-only fields are added later.
* **Firefox drag-and-drop broken on dataset browser**: File drops on the dataset browser page (`#/datasets/:name`) don't trigger in Firefox — the overlay never appears. Works fine in Chromium and works in the Add Files dialog (FileDropZone) in both browsers. A speculative `DOMStringList` fix was stashed but didn't resolve it. Needs real investigation. Stash: `wip: Firefox DOMStringList fix for dataset browser drag-and-drop`.

* **Migrate API validation errors to 422 Unprocessable Entity**: `validate_body` in `yadc/api/controllers/utils_json.py` currently raises `HTTPException` with status 400 for *all* Pydantic validation failures. The semantically cleaner split is 400 for *malformed* bodies (missing required fields, invalid enum values — the request itself is broken) and 422 for *semantically invalid* bodies (the body parsed cleanly but the values violate business rules — e.g. `new_name=""` after Pydantic parsing, caught downstream by the service). Discussed during the dataset-duplicate-plan; deferred for now since it's a cross-cutting change that touches every endpoint using `validate_body`, plus any client code that pattern-matches on the status code. Keep the duplicate endpoint on 400 to match the existing convention; if this ever gets done, it should be its own coordinated change with an audit of all callers and the frontend.
