---
name: gitignored-files
description: Gitignored paths in yadc (root .gitignore + yadc/webui/.gitignore). Read when unsure if a file is safe to touch, or when creating a new top-level directory.
category: convention
priority: 3
keep_updated: true
---

# Git-Ignored Files in yadc

The root `.gitignore` is **deliberately trimmed** to project-specific patterns. Do not reintroduce the default GitHub Python gitignore — its broad patterns (e.g. `env/`, `lib/`, `dist/`, `build/`) silently shadow project sub-folders (env store, webui `src/lib/`, etc.).

## Root `.gitignore` (project-specific)

| Pattern | What it ignores |
|---|---|
| `configs/*` + `!configs/example*` | all configs except the example |
| `templates/*` + `!templates/example*` | all templates except the example |
| `__pycache__/`, `*.py[cod]`, `*$py.class` | Python compilation |
| `*.so` | C extensions (defensive) |
| `*.egg-info/` | setuptools metadata (covers `yadc.egg-info/`) |
| `.pytest_cache/`, `.ruff_cache/` | test/lint cache |
| `.env`, `.envrc` | secrets |
| `.venv` | uv virtual env (uv default) |

## `yadc/webui/.gitignore` (frontend-specific)

SvelteKit, OS, env (with `!.env.example` / `!.env.test`), Vite artifacts. Note: the `!src/lib/` line is a now-redundant workaround for the root's dropped `lib/` rule — kept in place for now, harmless.

## Past collision history (don't reintroduce)

The root `.gitignore` previously inherited the GitHub Python template, which included broad patterns that silently shadowed project sub-folders:

- `env/` shadowed `yadc/webui/src/lib/stores/env/` — the env store was created on disk during a refactor but never tracked. Took a second such case (`lib/`) to surface and clean up.
- `lib/` shadowed `yadc/webui/src/lib/` — `yadc/webui/.gitignore` worked around it with `!src/lib/`.

If a new top-level sub-folder ever collides with a gitignore pattern, prefer to **remove the pattern** over adding a per-path negation. The cleanup itself is the prevention.

## Historical directories (not on disk, not gitignored)
These directories existed during development for testing/planned features but are no longer present:
- `datasets/` — dataset storage
- `output/` — training output
- `wandb/` — Weights & Biases logs
- `yadc/api_routes/` — experimental API routes module
- `yadc/captioners/extensions/` — experimental captioner extensions
- `yadc/captioners/hf_transformers/` — HuggingFace transformers support
- `yadc/training/` — training-related code
- `yadc/cmd/api/` — API command module
- `configs/tmp/` — temporary config files

## Auto-generated (always ignored)
- `__pycache__/` (everywhere)
- `.pytest_cache/`
- `*.egg-info/` / `yadc.egg-info/`

## Rule
When making changes, only modify tracked files. Do not create or edit ignored files unless explicitly asked by the user. When creating a new top-level directory, verify with `git check-ignore -v <path>` that it isn't being silently shadowed.
