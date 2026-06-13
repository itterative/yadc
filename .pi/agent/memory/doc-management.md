---
name: doc-management
description: Index of subsystem docs in docs/ and root-level reference memories. Use after architecture-overview to find detailed docs.
category: meta
priority: 1
keep_updated: true
---

# Doc Management

## Storage Location

All reference docs live in **`.pi/agent/memory/docs/`** (often referenced as `.pi/agent/memory/docs/` for brevity) inside the agent memory directory. These are **permanent reference documents** — architecture overviews, system descriptions, conventions, and workflow guides. They are separate from feature/design plans (which belong in `.pi/agent/memory/plans/`) and from the user-facing `docs/` directory at the repo root (which is for yadc end-users, not for the agent).

## Doc Index (in `.pi/agent/memory/docs/`)

| Doc | Description |
|-----|-------------|
| `backend/` | **Folder** of per-area structure files for the `yadc/` Python package. One file per area: `cli.md` (CLI entry points), `api.md` (Web UI backend), `cmd.md` (pure logic), `core.md` (business logic), `captioners.md` (API captioners). Each file has one-line summaries per file. Read the relevant area's doc on demand. |
| `frontend/` | **Folder** of per-area structure files for `yadc/webui/src/`. One file per area: `lib.md` (lib/ root modules), `stores.md` (lib/stores/), `components-ui.md` (lib/components/ui/), `components-domain.md` (lib/components/<domain>/), `routes.md` (routes/). Each file has one-line summaries per file. Read the relevant area's doc on demand. |
| [`api-di-system`](.pi/agent/memory/docs/api-di-system.md) | Web UI backend DI system — auto-discovery of services and controllers, injector binding lifecycle, and how to add new ones. |
| [`captioner-architecture`](.pi/agent/memory/docs/captioner-architecture.md) | Captioner hierarchy — `APICaptioner` auto-detection, inner captioner delegation, mixin pattern, streaming, stream error handling, and per-backend details. |
| [`dataset-watcher`](.pi/agent/memory/docs/dataset-watcher.md) | Filesystem watcher — inotify via watchdog, debouncing, expected-change tracking (`_expected_sources` + `_expected_files` + `_expected_patterns`), event dispatch, and frontend suppression. |
| [`dataset-system`](.pi/agent/memory/docs/dataset-system.md) | Dataset subsystem end-to-end — Web UI dataset model, three creation flows, managed dataset layout, upload pipeline (create/append/commit), source-id propagation, diff-scan rescan, background refresh, DatasetImage persistence. |
| [`captioning-workflow`](.pi/agent/memory/docs/captioning-workflow.md) | End-to-end captioning workflow — dataset loading, filtering, prediction loop, saving. |
| [`captioning-runner`](.pi/agent/memory/docs/captioning-runner.md) | Shared captioning runner (`yadc.core.captioning.CaptioningRunner`) used by both the CLI and the API. |
| [`cli-cmd-structure`](.pi/agent/memory/docs/cli-cmd-structure.md) | How CLI commands and `cmd/` modules are structured — click commands vs pure logic split. |
| [`debug-api-logging`](.pi/agent/memory/docs/debug-api-logging.md) | `YADC_DEBUG_CAPTION_RESPONSES=1` feature for logging caption API request/response pairs to JSONL files. |
| [`export-system`](.pi/agent/memory/docs/export-system.md) | How the export system works — backends, formats, and the draft/caption source selection. |
| [`frontend-patterns`](.pi/agent/memory/docs/frontend-patterns.md) | Frontend patterns — Abort contexts, Tabs system, Z-index layers, Topbar pattern, Browser notifications, Drop-to-upload, SSE, Svelte 5 conventions. |
| [`paths-and-storage`](.pi/agent/memory/docs/paths-and-storage.md) | File system paths used by yadc (platformdirs) and file storage conventions for `DatasetImage` persistence. |
| [`repository-pattern`](.pi/agent/memory/docs/repository-pattern.md) | Repository pattern for API services — repos own SQL + data model, services own transactions + business logic. Covers the `DBConnectionFactory.connection`/`transaction` contract, the auto-enrollment mechanism, and what belongs in which layer. |
| [`template-system`](.pi/agent/memory/docs/template-system.md) | Jinja2 prompt template system — template resolution, loading, and variable context. |
| [`yadc-config-v2`](.pi/agent/memory/docs/yadc-config-v2.md) | v2 dataset config format — `[[dataset]]` array-of-tables structure, v1 auto-conversion, strict/relaxed validation. |
| [`webui-frontend`](.pi/agent/memory/docs/webui-frontend.md) | yadc webui frontend setup — SvelteKit hash routing, Tailwind v4 configuration, Quart integration, and known issues. |
| [`codemirror-quirks`](.pi/agent/memory/docs/codemirror-quirks.md) | CodeMirror 6 editor sizing quirks — the CSS percentage-height trap, the flex/grid circular dependency, the `minmax(0, 1fr)` pattern that breaks it, and the absolute-positioning fallback kept in CodeMirror.svelte. |

## Root-Level Reference Memories

The following docs are kept at the `.pi/agent/memory/` root because they are consulted frequently across many tasks:

| Memory | Description |
|--------|-------------|
| `architecture-overview` | High-level project overview and doc index. Points to `backend/` and `frontend/` folders for file layout, and to focused docs (e.g. `dataset-system`, `captioner-architecture`) for subsystems. |
| `frontend-architecture` | Frontend organization rules (component placement, feature folders, store sub-folders) and styling patterns. Points to `frontend/` folder and `frontend-patterns`. |
| `dev-tools` | Dev tooling — ruff (linting & formatting) and basedpyright (type checking). |
| `git-conventions` | Git commit message conventions used in the yadc project. |
| `gitignored-files` | Files and directories ignored by git — must not be modified or included in changes. |
| `logging-format` | Log message format used across the yadc API — sentence-style message + optional `[key=value, ...]` block, positional `%`-formatting. |
| `pydantic-conventions` | Pydantic conventions used in the yadc project. |
| `running-python` | When running python in the yadc project, use `uv`. |
| `testing-conventions` | Test structure, patterns, and how to run tests in the yadc project. |
## When to Review Docs

**Before any of the following, read the relevant doc(s):**
- The user asks how a system or subsystem works (e.g. "how does the export system work?", "how are templates resolved?").
- The user asks about conventions or project rules (e.g. "what's our pydantic convention?", "how do I add a new CLI command?").
- The user is working in an area covered by a doc and may need architectural context.
- The user asks about file paths, storage locations, or config formats.

## Keeping Docs Up to Date

- **When a doc becomes obsolete**, move it to `.pi/agent/memory/docs/archive/` (create if needed) and remove it from the index.
- **When a new doc is created**, add it to the index with a concise description and decide whether it belongs in `docs/` or at the root.
- **When a doc's details change**, update the doc file itself and this index if the description needs updating.
- **Reference style**: use bare filenames (e.g. `api-di-system`) in section headings that declare the path, or full paths (e.g. `docs/api-di-system`) when the section is mid-document and the path isn't nearby. The path in question is `.pi/agent/memory/docs/` (often referred to as `.pi/agent/memory/docs/` for brevity to distinguish from the user-facing `docs/` at the repo root).
