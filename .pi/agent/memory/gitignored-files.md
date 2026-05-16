---
name: gitignored-files
description: Files and directories ignored by git in the yadc project. These must not be modified or included when making changes.
---

# Git-Ignored Files in yadc

## Project-specific ignored paths
- `configs/*` — all configs except `configs/example*` are gitignored
- `templates/*` — all templates except `templates/example*` are gitignored
- `datasets/` — dataset storage
- `output/` — training output
- `dist/`, `build/` — build artifacts
- `wandb/` — Weights & Biases logs
- `.venv/` — Python virtual environment
- `.env` — environment variables

## Ignored sub-directories (exist locally but not tracked)
- `yadc/api_routes/`
- `yadc/captioners/extensions/`
- `yadc/captioners/hf_transformers/`
- `yadc/training/`
- `yadc/cmd/api/`
- `configs/tmp/`

## Auto-generated (always ignored)
- `__pycache__/` (everywhere)
- `.pytest_cache/`
- `*.egg-info/` / `yadc.egg-info/`

## Rule
When making changes, only modify tracked files. Do not create or edit ignored files unless explicitly asked by the user.
