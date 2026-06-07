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

## Constructing services in tests

**Always use the real service constructor with mocked collaborators** — never `Service.__new__(Service)` to bypass construction. The project deliberately designs every service so its constructor can be called from a fixture: the collaborators a test does not care about take cheap `MagicMock(spec=...)` values, and the test asserts against `service.<attr>` for the real ones.

```python
@pytest.fixture
def service(self, test_configuration, logging_factory):
    from yadc.api.modules.dataset_watcher import DatasetWatcherService
    from yadc.api.modules.event_dispatcher import EventDispatcher

    return DatasetService(
        db=MagicMock(),
        watcher=MagicMock(spec=DatasetWatcherService),
        configuration=test_configuration,
        event_dispatcher=MagicMock(spec=EventDispatcher),
        logging=logging_factory,
        repo=mock_repo,  # the real mock the test asserts against
    )
```

If a service's constructor starts threads, creates sockets, or otherwise has side effects that prevent calling it from a test, **refactor the constructor** so the side effect moves to a lifecycle hook (e.g. a `@event_handler(StartupEvent)` method) or is injected. Do not work around it with `__new__`.

## Mocking with `patch()`

For tests that patch module-level imports in the code under test (e.g. `cmd_config`, `cmd_envs` imported into a controller), follow a three-step pattern that keeps the test refactor-friendly. Reference: `tests/api/test_envs.py`.

1. **Centralize patch target paths as module-level constants** so the import path lives in exactly one place. If the code under test renames an import, only the constant needs to change.
   ```python
   _PATCH_CMD_CONFIG = "yadc.api.controllers.api_envs.cmd_config"
   _PATCH_CMD_ENVS = "yadc.api.controllers.api_envs.cmd_envs"
   _PATCH_YADC_PASSWORD = "yadc.api.controllers.api_envs.YADC_PASSWORD"
   ```

2. **Wrap each patch target in a module-level fixture** that yields the mock. Tests take the fixture as a parameter instead of writing inline `with patch(...)` blocks. Prefer module-level over class-scoped fixtures so they're shared across test classes.
   ```python
   @pytest.fixture
   def patched_cmd_config():
       with patch(_PATCH_CMD_CONFIG) as mock_config:
           yield mock_config
   ```

3. **For patches where the value varies per test** (e.g. `YADC_PASSWORD` is `None` in one test, `"secret"` in another), use a factory fixture that returns a callable producing the `patch(...)` context manager:
   ```python
   @pytest.fixture
   def yadc_password():
       def _patch(value):
           return patch(_PATCH_YADC_PASSWORD, value)
       return _patch

   # Test usage:
   with yadc_password(None):
       ...
   ```

## API Endpoint Tests

Controller tests use a Quart test client fixture that registers the blueprint under test. See `tests/api/test_envs.py` for a representative example: a `client` fixture builds a Quart app, instantiates the blueprint with a mocked `LoggingFactory`, and returns `app.test_client()`. Group tests in classes by endpoint (e.g. `TestGetKeyMode`, `TestPutKeyMode`) with a one-line docstring describing what the endpoint does.
