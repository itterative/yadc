---
name: running-python
description: When a user query involves running Python scripts or commands in the yadc project, read this memory first.
---

You should use `uv run ...` when executing any python files in this project. Usual uv commands also apply, such as `uv sync`, `uv add ...`, etc.

**Important:** After changing Python version (e.g. `uv python pin 3.13`) or modifying dependencies, run `uv sync --all-extras` to ensure dev dependencies (pytest, ruff, etc.) are installed. Bare `uv sync` only syncs main dependencies.
