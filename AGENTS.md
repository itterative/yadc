You are working with yadc, a CLI tool for generating text captions for image datasets using vision-capable AI models. These captions are then exported in formats suitable for fine-tuning image generation models (e.g. sd-scripts).

## Commands

- **Run Python**: `uv run ...` (never bare `python`)
- **Lint**: `uv run ruff check yadc tests`
- **Format**: `uv run ruff format yadc tests`
- **Type check**: `uv run basedpyright <path>` (tests/ excluded from checking)
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

## Testing

- Mostly unit tests using `requests-mock` and Click's `CliRunner`
- Integration tests exist but require specific env setup — rarely needed
