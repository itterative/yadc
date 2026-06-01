---
name: testing-conventions
description: Test structure, patterns, and how to run tests. Read when writing or running tests.
category: workflow
priority: 3
---

# Testing Conventions

## Test Structure

Tests live under `tests/` and mirror `yadc/` structure (`api/`, `captioners/api/`, `cli/`, `core/`). Each subdirectory may have a `conftest.py` for shared fixtures. Individual test files follow the `test_*.py` naming convention.

`tests/api/conftest.py` provides shared fixtures for the API test suite:
- `test_configuration` — `Configuration` populated with paths under `tmp_path` (so tests don't touch real user data and the DB is recreated fresh for each test).
- `logging_factory` — `LoggingFactory` wired to the test configuration.
- `db_connection_factory` — a real `DBConnectionFactory` with a temp-file DB and migrations run synchronously.

Repository tests live next to their service tests under `tests/api/` (e.g. `test_dataset_repository.py`, `test_settings_repository.py`). Services are tested via in-memory mock repositories where possible, with a smaller number of integration tests using the real SQLite DB via the `db_connection_factory` fixture.

## Test Dependencies

In `pyproject.toml` under `[project.optional-dependencies].test`:
- `mock`, `pytest`, `pytest-integration`, `pytest-timeout`, `requests-mock`

## Running Tests

```bash
uv run pytest tests                    # all tests (integration tests excluded)
uv run pytest tests/core/              # core tests only
uv run pytest tests -k "test_name"     # specific test
uv run pytest -m "integration_test"    # run integration tests
```

Default timeout: 5 seconds. Integration tests are excluded by default via `addopts = "-m 'not integration_test'"` in `[tool.pytest.ini_options]`.

## Integration Tests

CLI tests that run the real `yadc` binary via subprocess (not Click's `CliRunner`) are marked with `@pytest.mark.integration_test` and skipped by default. Files:
- `tests/cli/test_cli_local.py` — local model servers (llamacpp, koboldcpp, vllm, ollama)
- `tests/cli/test_cli_official.py` — official API providers (gemini, openrouter, openai)
- `tests/cli/test_cli_envs.py` — env/keyring/config CLI commands

To run: `uv run pytest -m "integration_test"` (may skip individual tests if required envs are not configured).

## Test Patterns

- API captioner tests use `requests-mock` to mock HTTP responses
- Tests validate Pydantic response parsing, error handling
- Config tests cover v1→v2 conversion and validation edge cases
- Dataset resolver tests cover extras merging, inline image overrides
- CLI tests likely use Click's test runner (`CliRunner`)
