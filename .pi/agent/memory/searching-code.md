---
name: searching-code
description: When using grep or rg to search the codebase, read this memory for exclusion patterns and best practices.
---

# Searching Code with grep / rg

## Prefer `rg` (ripgrep) when available

`rg` automatically respects `.gitignore` and `.rgignore`, so most build/cache dirs are excluded out of the box. It is also faster.

```sh
rg "pattern" yadc/ tests/
```

## Using `grep`

Plain `grep -rn` will match everything, including build artifacts and caches. Exclude these directories by piping through `grep -v`:

```sh
grep -rn "pattern" yadc/ tests/ \
  | grep -v "__pycache__" \
  | grep -v ".svelte-kit" \
  | grep -v "node_modules" \
  | grep -v "build/" \
  | grep -v ".ruff_cache" \
  | grep -v ".pytest_cache" \
  | grep -v ".venv"
```

### Directories to exclude

| Directory            | Why                              |
|----------------------|----------------------------------|
| `__pycache__/`       | Python bytecode cache            |
| `node_modules/`      | JS/TS dependencies               |
| `.svelte-kit/`       | SvelteKit dev/build cache        |
| `build/`             | Build output (Python & frontend) |
| `.ruff_cache/`       | Ruff linting cache               |
| `.pytest_cache/`     | Pytest cache                     |
| `.venv/`             | Python virtual environment       |

## Tip: create a `.rgignore` for extra control

If `rg` still picks up unwanted files, add patterns to `.rgignore` at the project root. Current project does not have one.
