You are working with yadc, a CLI tool for generating text captions for image datasets using vision-capable AI models. These captions are then exported in formats suitable for fine-tuning image generation models (e.g. sd-scripts).

## Development Rules

1. When exploring or implementing changes, read any relevant memories you have. This process improves your efficiency.
2. Always give a thorough plan for your changes before performing them, unless the user asks for a change directly.
3. After you are finished with making changes requested by the user, look for any relevant memories you have and keep them up to date.
4. You are encouraged to ask the user any questions in order to figure out the best solution when given a task.

## Commands

### Python

```sh
uv run ... # always use uv, never bare python
uv run ruff check yadc tests
uv run ruff format yadc tests
uv run basedpyright <path>  # Python only — never on yadc/webui (use svelte-check for frontend)
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

### Comments

Comments are long-lived artifacts — calibrate them to long-term value, not the in-the-moment context you have while writing the code. Explain the **why** (non-obvious decisions, footguns, design trade-offs) — not the **what**, which the code already shows. Match the existing comment density: module docstrings in this codebase are typically 1–5 lines (the longest, `Application`, is ~15). If yours is much longer, you are probably over-explaining. Prefer trimming over adding — a comment that "seems useful right now" usually isn't, six months later, when the surrounding code has changed.

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

## Memories

Agent memories live under `.pi/agent/memory/`. Keep the two kinds separate:

- **Reference docs** (`docs/`, root-level reference memories like `architecture-overview`, and the plan index in `plan-management.md`) describe the **current** architecture in **present tense** and are long-term. Do **not** put phase/step markers ("Phase 7", "added in Phase 5b") or other transient implementation-tracking in them — update the description in place when things change. The plan index's Status column is the one exception: it is intentionally current-state, but keep it concise and high-level (e.g. "In Progress"), not a phase-by-phase enumeration.
- **Plans** (`plans/<plan>.md` + `plans/history/<plan>/`) are the **only** place for phased implementation tracking, design history, and done/remaining status. That churn belongs here, not in reference docs.

Rule of thumb: if a note will go stale after the next phase, it belongs in a plan, not a reference doc.

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
