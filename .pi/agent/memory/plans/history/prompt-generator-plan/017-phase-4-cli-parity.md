---
date: 2026-06-23
---

# Phase 4 — CLI Parity

## What was built

### Neutral `yadc/prompt_generation/` package (new)

The streaming core lives in a layer-agnostic package so the API
service and the CLI can both delegate to it without duplicating the
env-load → validate → client-build → stream → cleanup wiring:

- **`yadc/prompt_generation/models.py`** — `PromptGenerationRequest`,
  `ExamplePair`, `PromptGenerationFocus`, typed
  `PromptGenerationConfigError(PromptGenerationError, ValueError)`.
- **`yadc/prompt_generation/messages.py`** — system-prompt loading
  from `prompts/{generate,refine}.txt`, `_build_messages`,
  `_format_hints`, `_EXAMPLE_ACK` / `_REFINE_TEMPLATE_ACK`. Moved
  verbatim from the API service.
- **`yadc/prompt_generation/streaming.py`** —
  `stream_template_chunks(request, *, password, cache,
  response_logger, async_session) -> AsyncIterator[StreamChunk]`.
  Single source of truth for env-load → URL/model validation →
  AsyncSession + headers → `create_client` → message build →
  `predict_next_message_stream` → `aclose` finally.
- **`yadc/prompt_generation/__init__.py`** — re-exports the public
  surface (`ExamplePair`, `PromptGenerationRequest`,
  `PromptGenerationFocus`, `PromptGenerationError`,
  `PromptGenerationConfigError`, `stream_template_chunks`).
- **`yadc/prompt_generation/prompts/{generate,refine}.txt`** — moved
  (git mv) from `yadc/api/services/prompt_generation/prompts/`. The
  `[tool.setuptools.package-data]` entry in `pyproject.toml` was
  updated to point at the new location.

### Slimmed API service

- **`yadc/api/services/prompt_generation/service.py`** —
  `PromptGenerationService.generate(...)` is now a thin async-iterator
  wrapper that delegates to `stream_template_chunks` and forwards
  chunks (with one `Generating prompt template via env ...` log line
  up front). All the env-load / validation / client-build logic moved
  to the streaming core.
- **`yadc/api/services/prompt_generation/__init__.py`** — re-exports
  the models from `yadc.prompt_generation` so the existing import
  paths (`from yadc.api.services.prompt_generation import
  PromptGenerationRequest`) continue to work.

### CLI

- **`yadc/cmd/prompts/prompts.py`** — `generate(...)` is a
  CLI-friendly wrapper over `stream_template_chunks` that takes
  `on_chunk` + `on_reasoning` callbacks. Reads `os.environ` directly
  for `YADC_PASSWORD` (not the module constant) so callers that set
  the env var after import see the current value. Raises typed
  exceptions (`PromptGenerationConfigError`,
  `PasswordRequiredError`, `FileNotFoundError`) instead of calling
  `sys.exit`; the CLI maps them to exit codes. `--refine` resolves
  via `_resolve_refine_target`: if the arg matches an existing file,
  read it directly; otherwise look up `<name>.jinja` in the user
  template store via `cmd_templates.load_user_template`. A clear
  `FileNotFoundError` with a hint to run `yadc templates list`
  surfaces when neither matches.

  Few-shot examples are passed as positional `examples` arguments
  (files and/or directories). `resolve_examples_targets` (public —
  re-exported from `yadc/cmd/prompts/__init__.py`) handles the
  file/directory split: each image pairs with its `<stem>.txt`
  sidecar in the same directory; directories are scanned
  non-recursively in sorted order. `subject` is the filename stem
  (matches the webui's `ExamplesPanel.svelte` seed), `caption` is
  the trimmed sidecar, `image_data_url` is a base64 data URL with
  MIME type from `mimetypes.guess_type`. Image-encoding reuses no
  external dependency — just `pathlib + base64 + PIL via the
  existing `pillow` dep`. Missing sidecars / empty sidecars /
  non-image files all raise clear errors at the CLI level.
- **`yadc/cmd/prompts/__init__.py`** — re-exports `generate`.
- **`yadc/cli_prompts.py`** — click group + `generate` (with
  `--image-quality` `click.Choice(["auto", "low", "high"])` and
  `--refine` accepting a file path OR a user template name — no
  `click.Path(exists=True)` validation, so template names pass click
  and the cmd layer handles resolution) + variadic positional
  `EXAMPLES` argument (file paths and/or directories) + `--save-as
  <name>` (save the generated template to the user template store
  under this name) + `--force/--no-force` (overwrite existing
  template without prompting). `generate` parses args, resolves
  examples via `cmd_prompts.resolve_examples_targets`, runs the cmd
  `generate()` in `asyncio.run`, captures the streamed content, then
  optionally calls `_save_user_template(name, content, force=...)`.
  Catches the typed exceptions → maps to `STATUS_USER_ERROR` /
  `STATUS_ERROR`. `sys.exit` lives here, not in cmd. Reasoning is
  emitted once via `_logger.info` (quoted `> ` lines). Overwrite
  confirmation uses `click.confirm` in TTY mode; in non-TTY without
  `--force`, the command errors out. The `prompts save` subcommand
  was dropped (and never landed in main): piped LLM output to
  `<name>.jinja` skips the review step, and `--save-as` on
  `generate` gives the same end-state with the user watching the
  stream as the template is being saved.

### Tests

- **`tests/api/test_prompt_generation.py`** — patch target paths
  updated to the new module locations (`yadc.prompt_generation.streaming.cmd_envs`
  / `create_client`, `yadc.prompt_generation.messages._GENERATE_SYSTEM_PROMPT`
  / `_REFINE_SYSTEM_PROMPT`). Same assertions, same coverage. The
  `test_missing_api_url_raises_value_error` /
  `test_missing_model_name_raises_value_error` tests still pass
  because `PromptGenerationConfigError` is also a `ValueError`.
- **`tests/cli/test_cli_prompts.py`** — uses
  `cli(isolated=True)` from `tests/cli/conftest.py` (was a hand-rolled
  duplicate of the same fixture). Help + save tests run as real
  subprocess integration tests (no `uv run` shortcut, uses the
  conftest). A new `TestPromptsGenerateCmd` class exercises the cmd
  layer directly with the streaming core patched — chunk streaming,
  reasoning buffer + flush in finally (including the
  buffered-before-raise regression test), password resolution,
  `refine=path` reading from disk, typed-exception propagation.
- **`pyproject.toml`** — `[tool.setuptools.package-data]` updated.

### Plan / docs

- `.pi/agent/memory/plans/prompt-generator-plan.md` — Phase 4 history
  reference fixed (`016` → `017`), "Key Files" updated to reflect the
  new `yadc.prompt_generation/` package + the slimmed service.
- `.pi/agent/memory/docs/backend/cmd.md` — `prompts/` description
  updated.

## Design decisions

- **`yadc.prompt_generation` is a neutral package, not under cmd or
  api.** Both the CLI (`yadc.cmd.prompts`) and the API
  (`yadc.api.services.prompt_generation.PromptGenerationService`)
  delegate to it, so the streaming wiring lives in exactly one place.
  The dependency direction is clean: `api → prompt_generation → cmd /
  llm`, with no cycles. Extracted into its own package (not buried
  under cmd) so future entry points (e.g. a UI renderer, a different
  backend adapter) can build on the same domain models.

- **`PromptGenerationConfigError(PromptGenerationError, ValueError)`** —
  inherits from both so existing tests that expect `ValueError` keep
  working AND new code can catch the typed exception. Mirrors Python
  convention (bad config values are value errors).

- **CLI owns stdout / sys.exit; cmd raises typed exceptions.** The
  cmd `generate()` exposes `on_chunk` / `on_reasoning` callbacks so
  the CLI can route chunks to `sys.stdout` and reasoning to
  `_logger.info` without the cmd module touching either. This
  reverses the original draft's `sys.exit`-in-cmd pattern.

- **`os.environ.get("YADC_PASSWORD")` instead of the module
  constant.** The constant is captured at import time, which made
  `patch.dict("os.environ", {"YADC_PASSWORD": ...})` tests fail.
  Reading `os.environ` directly on each call picks up env changes
  (the cmd function is only called once per CLI invocation, so the
  no-op cost is fine).

- **`--save-as` over `prompts save` (or `--save` with optional
  value).** The original `prompts save <name>` subcommand was a
  thin wrapper around `cmd_templates.save_user_template` that
  read from stdin — it put the user template store behind a
  subcommand of `prompts` even though all other template CRUD
  already lives under `yadc templates`. More importantly, piping
  `prompts generate | prompts save` bypasses review: the user
  never sees the LLM output before it lands in the store. The
  replacement (`--save-as <name>` on `prompts generate`, plus
  `--force` for the overwrite case) keeps the streamed output
  visible while it generates, then writes the same bytes to the
  template store as a single command. The interactive-prompt
  variant (TTY-mode `--save` with no value) was considered and
  rejected: click 8.3 doesn't natively support optional-value
  options for non-flag types (`is_flag=False` requires a value,
  `is_flag=True` rejects a value), and the custom `click.Option`
  subclass / `prompt=True` workarounds either still leak
  `Aborted!` in non-TTY or add non-trivial complexity for a
  UX-nicety. If the user later wants an interactive prompt
  workflow, `--save-as` plus `yadc templates edit <name>` covers
  the same ground.

- **`on_chunk` / `on_reasoning` callbacks over a sink object.** Two
  callbacks are simpler than a single sink-with-event-types; the
  reasoning can never accidentally interleave with text on the same
  callback, and the default no-op implementation is one function.

- **Reasoning buffered + emitted-once.** Reasoning chunks arrive in
  arbitrary order relative to text. The cmd layer buffers them and
  emits via `on_reasoning` once — either on the first text chunk
  (via `_emit_reasoning`) or in `finally` if the stream raises /
  ends without text. The `finally` flush is what the existing draft
  was missing.

## Files changed

- `yadc/prompt_generation/{__init__,models,messages,streaming}.py` —
  new
- `yadc/prompt_generation/prompts/{generate,refine}.txt` — moved from
  `yadc/api/services/prompt_generation/prompts/`
- `yadc/api/services/prompt_generation/{__init__,service}.py` —
  slimmed
- `yadc/cmd/prompts/{__init__,prompts}.py` — new (rewrite)
- `yadc/cli_prompts.py` — new (rewrite)
- `yadc/cli.py` — `cli_prompts` import + registration (unchanged from
  first draft)
- `tests/cli/test_cli_prompts.py` — rewrite: shared fixture +
  `TestPromptsGenerateCmd` unit tests
- `tests/api/test_prompt_generation.py` — patch paths updated
- `pyproject.toml` — `[tool.setuptools.package-data]` updated
- `.pi/agent/memory/plans/prompt-generator-plan.md` — Phase 4 history
  ref, Key Files
- `.pi/agent/memory/docs/backend/cmd.md` — `prompts/` description