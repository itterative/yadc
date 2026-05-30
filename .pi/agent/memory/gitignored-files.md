---
name: gitignored-files
description: Files and directories that must not be modified or included in changes. Read when unsure if a file is safe to touch.
category: convention
priority: 3
---

# Git-Ignored Files in yadc

## Project-specific gitignored paths
- `configs/*` — all configs except `configs/example*` are gitignored
- `templates/*` — all templates except `templates/example*` are gitignored
- `dist/`, `build/` — build artifacts (from Python default gitignore)
- `.venv/` — Python virtual environment
- `.env` — environment variables
- `yadc/webui/node_modules/`, `yadc/webui/.svelte-kit/`, `yadc/webui/build/` — frontend build artifacts

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
When making changes, only modify tracked files. Do not create or edit ignored files unless explicitly asked by the user.
