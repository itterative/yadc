---
name: testing-conventions
description: When a user query involves tests, test structure, or how to run tests in the yadc project, read this memory first.
---

# Testing Conventions

## Test Structure

```
tests/
  conftest.py                    # shared fixtures
  test_cli_draft.py              # draft CLI tests

  captioners/api/
    conftest.py                  # API test fixtures
    test_gemini.py
    test_koboldcpp.py
    test_llamacpp.py
    test_openai.py
    test_openrouter.py

  cli/
    conftest.py                  # CLI test fixtures
    test_cli_local.py            # local API tests
    test_cli_official.py         # official API tests (OpenAI, Gemini, etc.)

  core/
    test_config.py               # Config parsing (v1/v2)
    test_dataset_resolver.py     # Dataset resolution, extras merging
    test_export.py               # Export functionality
```

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
