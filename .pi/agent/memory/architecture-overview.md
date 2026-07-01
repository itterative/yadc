---
name: architecture-overview
description: High-level project structure and cross-cutting conventions. Start here for structural work or cross-cutting changes. For file-by-file structure, see `backend/` and `frontend/` folders.
category: architecture
priority: 1
keep_updated: true
---

# yadc Architecture Overview

## What is yadc

Yet Another Dataset Captioner — a CLI tool for captioning image datasets using remote AI APIs (OpenAI, Gemini, OpenRouter, vLLM, llama.cpp, koboldcpp, Ollama).

## See also (in `.pi/agent/memory/docs/`)

- `backend/` — folder of per-area structure files for the `yadc/` Python package (CLI, `api/`, `cmd/`, `core/`, `captioners/`, `templates/`, `utils/`)
- `frontend/` — folder of per-area structure files for `yadc/webui/src/` (`lib/`, `styles/`, `stores/`, `components-ui/`, `components-domain/`, `routes/`)
- `api-di-system` — Quart + injector DI auto-discovery
- `repository-pattern` — service/repo split for SQL
- `captioner-architecture` — single `APICaptioner` composes a `BaseLLMClient`; mixin/helpers, conversation building, streaming error handling.
- `backend/llm` (in `docs/backend/`) — backend-agnostic LLM client layer (`yadc/llm/`); self-contained (shared HTTP infra lives here); composed by the captioner and used directly by the prompt generator
- `dataset-watcher` — inotify watcher + expected-change tracking + event suppression
- `dataset-system` — dataset model, creation flows, upload pipeline, rescan, persistence
- `paths-and-storage` — platformdirs paths + DatasetImage file conventions
- `template-system` — Jinja2 prompt template resolution
- `yadc-config-v2` — v2 dataset config format + strict/relaxed validation
- `cli-cmd-structure` — Click CLI vs `cmd/` pure-logic split
- `captioning-workflow` — end-to-end captioning flow
- `dataset-job-coordinator-plan` — `DatasetJobService` cross-service per-dataset job mutex (captioning vs tagging)
- `captioning-runner` — shared captioning runner (CLI + API)
- `export-system` — export backends + format
- `debug-api-logging` — `YADC_DEBUG_CAPTION_RESPONSES=1` debug logging
- `frontend-architecture` — frontend rules (placement, organization)
- `frontend-patterns` — frontend patterns (Tabs, Z-index, Topbar, Notifications, Drop-to-upload, SSE)
- `webui-frontend` — SvelteKit/Tailwind v4 setup
- `codemirror-quirks` — CodeMirror 6 sizing pitfalls

## Top-level non-package directories

- `scripts/` — standalone, not part of the `yadc/` importable package. Contains the desktop-.exe tooling: `entrypoints/webui_desktop.py` (PyInstaller entry point — mutates `sys.argv` then dispatches to `yadc.cli:cli`), `build_exe.spec` / `build_exe_gpu.spec` (PyInstaller specs), and `build_windows_exe.{ps1,sh}` (build drivers). See `plans/desktop-exe-plan` and `docs/desktop-exe.md`. CI build/publish lives in `.github/workflows/release-assets.yml`.
