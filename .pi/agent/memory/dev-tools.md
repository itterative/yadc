---
name: dev-tools
description: Dev tooling — ruff (linting & formatting) and basedpyright (type checking).
---

Both are dev dependencies installed via `uv add --dev`.

## Ruff (linter & formatter)

Config in `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.lint]`.

- Target: Python 3.11
- Line length: 160
- Rule sets: E, F, W, I (pycodestyle, pyflakes, warnings, isort)
- `__init__.py` re-exports use `__all__` lists to avoid F401 false positives

Run with: `uv run ruff check yadc tests` and `uv run ruff format yadc tests`.

## Basedpyright (type checker)

Run with: `uv run basedpyright <path>`

Config in `pyproject.toml` under `[tool.basedpyright]`.
Excludes: `tests/`, `.venv/`.
Several noisy strict-mode rules are disabled globally (unknown types, import cycles, private usage, etc.).
Remaining real issues are fixed: `@override` decorators, `X | None` instead of `Optional`, `collections.abc.Generator`, `dict[str, Any]` for JSON payloads, match exhaustiveness.
