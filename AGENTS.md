You are working with yadc, a CLI tool for generating text captions for image datasets using vision-capable AI models. These captions are then exported in formats suitable for fine-tuning image generation models (e.g. sd-scripts).

## Commands

- **Run Python**: `uv run ...` (never bare `python`)

### Python

```sh
uv run ruff check yadc tests
uv run ruff format yadc tests
uv run basedpyright <path>  # tests/ excluded from checking
```

### Frontend

```sh
cd yadc/webui
npx eslint .
npx eslint --fix .  # run before prettier
npx prettier --write .
npx svelte-check --tsconfig ./tsconfig.json
npm run build
```

### Tests

```sh
uv run pytest tests
uv run pytest tests -k "test_name"
```

## Code Style

### Python

- **Ruff**: line-length 160, rules E/F/W/I (pycodestyle, pyflakes, warnings, isort)
- **Basedpyright**: several strict-mode rules disabled globally (see `[tool.basedpyright]` in pyproject.toml)

### Frontend

- **Prettier**: configured in `yadc/webui/.prettierrc` — plugins: `prettier-plugin-svelte`, `prettier-plugin-tailwindcss`
- **ESLint**: `curly: ['error', 'all']` — always require braces on if/else/for/while blocks
- **Workflow**: run `eslint --fix .` then `prettier --write .` — ESLint adds braces, Prettier expands them to multi-line with proper formatting

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

```sh
tmux new-session -d -s yadc-webui "uv run yadc webui serve --host 127.0.0.1"
tmux kill-session -t yadc-webui
tmux has-session -t yadc-webui 2>/dev/null
tmux capture-pane -t yadc-webui -p | tail -30  # can pipe through head/tail/grep
tmux list-sessions
```

## Testing

- Mostly unit tests using `requests-mock` and Click's `CliRunner`
- Integration tests exist but require specific env setup — rarely needed
