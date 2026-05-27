---
name: testing-conventions
description: When a user query involves tests, test structure, or how to run tests in the yadc project, read this memory first.
---

# Testing Conventions

## Test Structure

Tests live under `tests/` and mirror `yadc/` structure (`api/`, `captioners/api/`, `cli/`, `core/`). Each subdirectory may have a `conftest.py` for shared fixtures. Individual test files follow the `test_*.py` naming convention.

## Test Dependencies

In `pyproject.toml` under `[project.optional-dependencies].test`:
- `mock`, `pytest`, `pytest-integration`, `pytest-timeout`, `requests-mock`

## Running Tests

```bash
uv run pytest tests                    # all tests
uv run pytest tests/core/              # core tests only
uv run pytest tests -k "test_name"     # specific test
```

Default timeout: 5 seconds (configured in `[tool.pytest.ini_options]`).

## Test Patterns

- API captioner tests use `requests-mock` to mock HTTP responses
- Tests validate Pydantic response parsing, error handling
- Config tests cover v1→v2 conversion and validation edge cases
- Dataset resolver tests cover extras merging, inline image overrides
- CLI tests likely use Click's test runner (`CliRunner`)
