---
name: doc-management
description: Index of subsystem docs in docs/ and root-level reference memories. Use after architecture-overview to find detailed docs.
category: meta
priority: 1
keep_updated: true
---

# Doc Management

## Storage Location

All reference docs live in **`.pi/agent/memory/docs/`** inside the agent memory directory. These are **permanent reference documents** — architecture overviews, system descriptions, conventions, and workflow guides. They are separate from feature/design plans (which belong in `.pi/agent/memory/plans/`).

## Doc Index

| Doc | Description |
|-----|-------------|
| [`api-di-system`](.pi/agent/memory/docs/api-di-system.md) | Web UI backend DI system — auto-discovery of services and controllers, injector binding lifecycle, and how to add new ones. |
| [`captioner-architecture`](.pi/agent/memory/docs/captioner-architecture.md) | Captioner hierarchy — `APICaptioner` auto-detection, inner captioner delegation, mixin pattern, and per-backend details. |
| [`captioning-workflow`](.pi/agent/memory/docs/captioning-workflow.md) | End-to-end captioning workflow — dataset loading, filtering, prediction loop, saving. |
| [`cli-cmd-structure`](.pi/agent/memory/docs/cli-cmd-structure.md) | How CLI commands and `cmd/` modules are structured — click commands vs pure logic split. |
| [`debug-api-logging`](.pi/agent/memory/docs/debug-api-logging.md) | `YADC_DEBUG_CAPTION_RESPONSES=1` feature for logging caption API request/response pairs to JSONL files. |
| [`export-system`](.pi/agent/memory/docs/export-system.md) | How the export system works — backends, formats, and the draft/caption source selection. |
| [`paths-and-storage`](.pi/agent/memory/docs/paths-and-storage.md) | File system paths used by yadc (platformdirs) and file storage conventions for `DatasetImage` persistence. |
| [`template-system`](.pi/agent/memory/docs/template-system.md) | Jinja2 prompt template system — template resolution, loading, and variable context. |
| [`yadc-config-v2`](.pi/agent/memory/docs/yadc-config-v2.md) | v2 dataset config format — `[[dataset]]` array-of-tables structure and v1 auto-conversion. |
| [`webui-frontend`](.pi/agent/memory/docs/webui-frontend.md) | yadc webui frontend setup — SvelteKit hash routing, Tailwind v4 configuration, Quart integration, and known issues. |

## Root-Level Reference Memories

The following docs are kept at the `.pi/agent/memory/` root because they are consulted frequently across many tasks:

| Memory | Description |
|--------|-------------|
| `architecture-overview` | High-level project architecture and module organization. |
| `frontend-architecture` | Frontend directory structure, stores, components, routes, and key patterns. |
| `dev-tools` | Dev tooling — ruff (linting & formatting) and basedpyright (type checking). |
| `git-conventions` | Git commit message conventions used in the yadc project. |
| `gitignored-files` | Files and directories ignored by git — must not be modified or included in changes. |
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
- **Reference style**: use `docs/<name>` in other memories for docs in the docs directory.
