You are working with yadc, a CLI tool for generating text captions for image datasets using vision-capable AI models. These captions are then exported in formats suitable for fine-tuning image generation models (e.g. sd-scripts).

## Commands

- **Run Python**: `uv run ...` (never bare `python`)
- **Lint**: `uv run ruff check yadc tests` *(only when changing Python files)*
- **Format**: `uv run ruff format yadc tests` *(only when changing Python files)*
- **Type check**: `uv run basedpyright <path>` *(only when changing Python files; tests/ excluded from checking)*
- **Frontend lint**: `cd yadc/webui && npx eslint .` *(only when changing frontend files)*
- **Frontend check**: `cd yadc/webui && npx svelte-check --tsconfig ./tsconfig.json` *(only when changing frontend files)*
- **Frontend build**: `cd yadc/webui && npm run build` *(only when changing frontend files)*
- **Test**: `uv run pytest tests` — 5s timeout per test (configured in pyproject.toml)
- **Single test**: `uv run pytest tests -k "test_name"`

## Code Style

- **Ruff**: line-length 160, rules E/F/W/I (pycodestyle, pyflakes, warnings, isort)
- **Basedpyright**: several strict-mode rules disabled globally (see `[tool.basedpyright]` in pyproject.toml)

### Pydantic

- Use `Model.model_validate(data)` instead of `Model(**data)` when deserializing from dicts — `**` bypasses union resolution
- `model_config` goes at the **end** of the class body, annotated as `ClassVar[ConfigDict]`

## Architecture Conventions

- **CLI/cmd split**: `yadc/cli_<name>.py` has click commands; `yadc/cmd/<name>/` has pure logic (no click imports). CLI imports the cmd package, not the submodule.
- **`__init__.py` re-exports**: use `__all__` lists to avoid ruff F401 false positives
- **Conventional commits**: `type: subject` or `type(scope): subject` (feat, fix, refactor, docs)

## Key Paths

- **Python package**: `yadc/` (CLI, core logic, API backend, webui build output)
- **WebUI source**: `yadc/webui/` (SvelteKit frontend — NOT at repo root)
- **Tests**: `tests/` (mirrors `yadc/` structure: `cli/`, `captioners/`, `core/`)
- **Project config**: `pyproject.toml` at repo root

## WebUI (Development)

Run the webui server in a tmux session so it stays in the background.
Must use **single tmux commands** (no chaining) due to bash sandbox rules.

### Allowed tmux commands

- **Start**: `tmux new-session -d -s yadc-webui "uv run yadc webui serve --host 127.0.0.1"`
- **Stop**: `tmux kill-session -t yadc-webui`
- **Check if running**: `tmux has-session -t yadc-webui 2>/dev/null`
- **Capture output**: `tmux capture-pane -t yadc-webui -p | tail -30` (grab only the last 20-30 lines to see recent logs/errors)
- **Pipe capture**: `| head *`, `| tail *`, and `| grep *` are allowed on capture-pane output
- **List sessions**: `tmux list-sessions`

Use this whenever you need a live server for testing webui changes.

## Testing

- Mostly unit tests using `requests-mock` and Click's `CliRunner`
- Integration tests exist but require specific env setup — rarely needed
