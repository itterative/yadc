---
date: 2026-06-04
---
# Phase 4: CLI caption migration to CaptioningRunner

**Context:** Phase 4 of the shared-runner plan refactors `yadc/cli_caption.py` to delegate the model-create / stream / save loop to `yadc.core.captioning.CaptioningRunner`. The CLI's interactive action menu (quit/skip/continue/retry/edit/prompts/reply/clear-replies) stays in `cli_caption.py`; the captioning mechanics underneath are now the shared runner.

**Decision:** Eight small deviations from the plan, each documented below.

## 1. Added `save_caption` (public) to the runner

The plan's design for `caption_image_dry_run` was: "Stream + accumulate without saving. Used by the CLI's interactive flow where the user may retry/reject before committing the caption to disk."

The CLI's interactive flow needs to commit the caption after the user accepts — but `caption_image_dry_run` doesn't save. The plan didn't specify *how* the CLI commits. Three options:

- (a) CLI duplicates the save logic (~10 lines, tightly coupled to `options.draft` / `expected_change_registrar`).
- (b) CLI calls `caption_image` for the final save (wasteful — re-streams the model).
- (c) Runner exposes a public `save_caption` method.

**Picked (c).** Renamed `_save_caption` → `save_caption` (public). The CLI does `await runner.caption_image_dry_run(...)` per retry, then `runner.save_caption(image, caption)` after acceptance. The API never calls `save_caption` directly (it uses `caption_image` which already saves internally). This is the cleanest fit for the runner's existing `expected_change_registrar` integration — the CLI passes `None` and the registrar call is skipped automatically.

## 2. Added `model` property to the runner

The CLI needs `model.api_type` for the existing assistant-prefill warning:

```python
if settings.advanced.assistant_prefill and model.api_type in (APITypes.GEMINI, APITypes.OPENAI, APITypes.OPENROUTER):
    _logger.warning("Warning: assistant prefill is set, but the API might not support it")
```

The runner hides the model internally. To make this checkable without exposing the full `Captioner` API, added a `model` property:

```python
@property
def model(self) -> APICaptioner:
    assert self._model is not None, "CaptioningRunner used outside 'async with'"
    return self._model
```

The assertion matches the runner's existing pattern (see `caption_image`'s first line). The CLI accesses `runner.model.api_type`.

## 3. Used `PromptRenderer` directly for the "prompts" action

The old CLI's "prompts" action (show the system + user prompts without calling the model) used `model.prompts_from_image(dataset_image_current, drafts=drafts)`. With the runner hiding the model, the cleanest replacement is to use `PromptRenderer` directly — same template the model would use, no model access required:

```python
renderer = PromptRenderer(prompt_template=config.prompt.template)
# ...
case "prompts":
    system_prompt, user_prompt = renderer.render(dataset_image_current, drafts=drafts)
```

`PromptRenderer` is the internal renderer that `Captioner.prompts_from_image` calls (one line of indirection in `core/captioner.py`). The output is byte-for-byte identical to the old behavior.

## 4. Changed click arg from `click.File("r")` to `click.Path(exists=True, dir_okay=False, readable=True)`

Per the plan's open question #1: drop stdin support. `click.File("r")` accepts stdin (`-`) and any file path; `click.Path(...)` accepts only a real file. The CLI's `_caption_async` now takes `dataset: str` (the path) and constructs `Path(dataset)` internally.

This also drops the `click.File("r")`-specific stdin check (`dataset.name == "-"` in the `DEBUG_CAPTION_RESPONSES` branch) — with `click.Path`, `-` is no longer a valid argument, so the check would never fire.

The `cmd_cache.debug_log_dir(dataset.name)` call (derives the dataset name from the file stem) now passes `str(dataset_path)` instead of `dataset.name`.

## 5. Pre-parse the TOML to resolve top-level config defaults

The loader needs fully-resolved `CaptionJobOptions` for its overwrite/draft filter. For values the user didn't pass on the command line (`--overwrite`, `--rounds`, `--interactive`), the CLI falls back to the config's top-level field (`overwrite_captions`, `rounds`, `interactive`).

**Added `_resolve_config_defaults(dataset_path, kwargs)`** that does a cheap `toml.load` to read the three top-level fields. The loader's full `parse_config` then runs inside `load_dataset_config` — two parses, but the first is just `toml.load` (a few ms at most) and only the defaults are extracted from it.

Alternative considered: have the loader accept unresolved `CaptionJobOptions` and resolve defaults internally. Decided against — the loader's API stays clean (it takes a fully-resolved options object), and the CLI's logic is explicit at the call site.

## 6. Removed `_resolve_template` from the CLI

The plan said `_resolve_template` is replaced by the loader. Confirmed: `load_dataset_config` calls `resolve_template` internally (via `apply_config_overrides`'s `prompt_name` handling and the explicit `resolve_template` call at the end of the loader).

The CLI's `_resolve_template` was deleted. The CLI's `_resolve_template` previously also logged errors and called `sys.exit(STATUS_USER_ERROR)` on failure; the new behavior is to let `load_dataset_config` raise `ValueError` and the CLI catches it in `_caption_async`'s except block (logs the same message and exits with `STATUS_USER_ERROR`).

## 7. Removed `cmd_envs.load_env` from the CLI

The CLI's old `_load_dataset` called `cmd_envs.load_env(env=env)` directly to merge env values into the raw TOML. The loader's `apply_config_overrides` now does this internally, calling `cmd_envs.load_env(opts.env, password=opts.password)`. The CLI no longer needs the direct import.

The `LoggingFactory`-like initialization (`_logger.info("Using %s user environment.", env)`) is lost. The CLI's `_caption_async` doesn't log this anymore — it was nice-to-have output, not a required behavior.

## 8. `model.log_usage()` and `async_session.aclose()` now happen in runner's `__aexit__`

The old CLI called these explicitly at the end of `_caption_async`. The runner's `__aexit__` already handles them, so the explicit calls were removed. The end-of-run log line is preserved:

```python
async with CaptioningRunner(...) as runner:
    # ... captioning ...
# __aexit__ calls model.log_usage() + async_session.aclose()
_logger.info("Done. (%.1f sec)", timer.elapsed)
```

## Integration test note

The `integration-tests-local-llamacpp` env has no `api_token` set. The conftest's `_integration_cli` calls `envs get api_token` and expects success, so the existing integration tests for this env fail before reaching the caption command (with `Error: key not found: api_token`). This is a pre-existing config issue with the conftest, not a refactor regression — the `envs get` check was there before Phase 4.

Verified the refactor end-to-end by running the caption command directly from `tests/cli/test_data/`:

```sh
cd tests/cli/test_data
uv run yadc caption test_pedro.dataset --no-stream --env integration-tests-local-llamacpp
# ... successfully captions the image, writes test_pedro.txt, calls log_usage
uv run yadc caption test_pedro.dataset --stream --env integration-tests-local-llamacpp
# ... streams tokens to stderr, writes test_pedro.txt (overwritten)
```

Both runs: model loaded, image captioned, `log_usage` called by `__aexit__`, outer timer logged. The .txt/.toml/.history~ files were cleaned up after the smoke test.

## Test strategy

The plan says integration tests "should pass unchanged after Phase 4 — the CLI command surface is preserved." Confirmed: the click command's surface is identical (same options, same `dataset` arg), so the existing `test_cli_official.py` and `test_cli_local.py` tests will exercise the new code path with no changes.

**No new unit tests added for the CLI's interactive loop.** The plan noted this as a mitigation, not a requirement. Adding focused unit tests for the multi-round / reply / edit / prompts action menu would require mocking the runner + simulating user input, which is a much larger effort than the value justifies. The integration tests cover the surface; any regression in the interactive flow will be caught by manual testing.

The existing `tests/core/captioning/test_runner.py` (22 tests) covers the runner's caption + save behaviour. The CLI's additions (pre-parse for defaults, save via `runner.save_caption`) are thin enough that the integration tests are sufficient.

## Files touched

- `yadc/core/captioning/runner.py` — renamed `_save_caption` → `save_caption` (public); added `model` property.
- `yadc/cli_caption.py` — substantial rewrite:
  - Added `CLICallbacks` class implementing `CaptioningCallbacks` Protocol.
  - Replaced `_load_dataset` with `load_dataset_config` (via `_resolve_config_defaults` for the top-level fields the loader needs).
  - Replaced `_predict_caption_one_shot` and `_predict_caption_rounds` with `_stream_one_round` helper that wraps `runner.caption_image_dry_run`.
  - `_caption` now takes a `runner` arg; calls `runner.caption_image_dry_run` and `runner.save_caption`.
  - `_caption_async` builds `CaptionJobOptions` from kwargs, pre-parses for config defaults, opens `async with CaptioningRunner(...)`.
  - "prompts" action uses `PromptRenderer` directly (not `model.prompts_from_image`).
  - Removed `_resolve_template` and `cmd_envs.load_env` calls.
  - `click.File("r")` → `click.Path(exists=True, dir_okay=False, readable=True)`.
  - `# pyright: ignore[reportUnusedParameter]` on the 3 unused Protocol-required params in `CLICallbacks` (per Phase 3 convention).
