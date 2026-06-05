---
name: cli-api-captioning-shared-runner-plan
description: Share a CaptioningRunner core between the CLI's `yadc caption` command and the API's `AsyncCaptionJob` — both currently re-implement the same model-create / stream / save loop. First step toward unifying the CLI and API captioning flows.
status: Approved (with decisions)
decisions:
  shared_core_location: yadc/core/captioning/
  cli_stdin: drop
  cli_di: none
  logging_refactor: deferred_phase (decisions pending after runner implementation)
---

# Shared Captioning Runner — CLI ↔ API

## Problem

`yadc/cli_caption.py` and `yadc/api/services/captioning.py::AsyncCaptionJob` re-implement the same captioning loop. The duplication has already drifted in a few places and is the main obstacle to keeping CLI and WebUI behaviour in sync.

### What's duplicated today

| Step | CLI | API |
|------|-----|-----|
| Load + parse dataset TOML | `cli_caption._load_dataset` (inline) | `AsyncCaptionJob.preflight_images` + re-parse in `_ado_run` |
| Merge env / CLI / TOML overrides into raw config | inline in `_load_dataset` | `apply_config_overrides()` |
| Resolve prompt template (user → builtin → default) | `_resolve_template` (inline) | `resolve_template()` (free function) |
| Create `APICaptioner` + `AsyncSession` | inline in `_caption_async` | inline in `_ado_run` |
| Filter images (skip already-captioned unless `--overwrite`) | inline in `_caption_async` | inline in `preflight_images` |
| Stream token-by-token, accumulate, save caption | `_predict_caption_one_shot` / `_predict_caption_rounds` | `_acaption_one` |
| `model.log_usage()` + `async_session.aclose()` | inline at the end of `_caption_async` | inline at the end of `_ado_run` |

The two implementations also differ in defaults and surrounding behaviour:

- **CLI** has `--interactive`, multi-round, retry/edit/prompts/reply, draft files, response-debug logging, HTTP response cache, `click`-based output.
- **API** has per-dataset jobs, SSE event emission, watcher `expect_file_change()` registration, job cancellation via `asyncio.Event`, single-image mode (`image_ids`).

The user's first refactor goal is to make the CLI use the API's captioning service. Doing that naively (CLI calls the live `CaptioningService` over HTTP, or constructs the entire DI container in-process) is the wrong shape — the service is tied to the SQLite-backed `DatasetService`, the DI-managed `EventDispatcher`, the `DatasetWatcherService`, and SSE-style events. The interactive flow doesn't fit a job-with-events model at all.

The right shape is to **extract the loop into a shared core** that both the CLI and the API use, with each side plugging in its own callbacks for output / event emission / cancellation.

## Goal

Extract a pure `CaptioningRunner` (no DI, no Flask/Quart, no click) that owns the model-create / stream / save loop. Both `cli_caption.py` and `AsyncCaptionJob` become thin orchestrators that build a `CaptioningRunner` and pass it callbacks.

This is the foundation for the broader CLI/API unification (mentioned in the user's request) — once both sides share the runner, further consolidation (e.g. the API hosting its own CLI subcommands, or the CLI being a thin client over the API) becomes a much smaller delta.

## Design

### New package: `yadc/core/captioning/`

A new sub-package under `yadc/core/` (next to `captioner.py`, `dataset.py`, etc.). The runner is core business logic; both `yadc/api/` and `yadc/cli_*.py` import from it.

```
yadc/core/captioning/
  __init__.py          # re-exports public API
  options.py           # CaptionJobOptions (Pydantic model, moved from API)
  loader.py            # apply_config_overrides, resolve_template, load_dataset_config
  runner.py            # CaptioningRunner + CaptioningCallbacks + CaptioningStatus
```

`core/` is the right place: it already contains `captioner.py` (HTTP-using base class) and `dataset_resolver.py` (shared between CLI and API). No circular-import risk: nothing under `core/` imports from `api/`, `cli_*.py`, or `cmd/`.

### `options.py` — `CaptionJobOptions`

Move the existing Pydantic model from `yadc/api/services/captioning.py` to `yadc/core/captioning/options.py` verbatim. This model is the canonical "all knobs for one captioning run" object — both the API's JSON body and the CLI's click kwargs can be coerced into it.

Re-export from `yadc/api/services/captioning.py` for backward compatibility:
```python
# yadc/api/services/captioning.py
from yadc.core.captioning import CaptionJobOptions  # re-export
```

### `loader.py` — config loading + overrides + template resolution

Move the two existing free functions from the API and add a new top-level helper:

```python
# yadc/core/captioning/loader.py
def apply_config_overrides(raw: dict[str, Any], opts: CaptionJobOptions) -> dict[str, Any]: ...
def resolve_template(prompt_name: str, prompt_template: str, logger: Logger | None = None) -> str: ...

def load_dataset_config(
    config_path: str | Path,
    options: CaptionJobOptions,
    *,
    user_config: str | None = None,  # CLI-only: merge with named user config
    base_dir: str | Path | None = None,  # for relative paths; defaults to config_path.parent
) -> tuple[Config, list[DatasetImage]]:
    """Load TOML, apply options, parse Config, resolve images.

    Returns (Config, list_of_DatasetImage). The list is already filtered
    for overwrite/draft (the runner is responsible only for iteration).
    """
```

The CLI's `_load_dataset()` and the API's `preflight_images()` + the second re-parse in `_ado_run()` all collapse into a single call to `load_dataset_config()`.

Re-export the moved functions from the API for backward compatibility.

### `runner.py` — `CaptioningRunner` + `CaptioningCallbacks`

```python
@dataclass
class CaptioningCallbacks:
    """Optional callbacks. All fields default to None (no-op)."""
    on_token: Callable[[str], None] | None = None                       # per streamed token
    on_image_started: Callable[[DatasetImage], None] | None = None      # before model call
    on_image_captioned: Callable[[DatasetImage, int], None] | None = None  # (image, duration_ms) after save
    on_image_error: Callable[[DatasetImage, str, int], None] | None = None   # (image, error_msg, duration_ms)
    on_status: Callable[[CaptioningStatus], None] | None = None         # batch status changes
    on_usage: Callable[[], None] | None = None                          # model.log_usage() should be called


@dataclass
class CaptioningStatus:
    status: Literal["running", "done", "cancelled", "error"]
    processed: int = 0
    total: int = 0
    errors: int = 0


class CaptioningRunner:
    """Pure async captioning loop. No DI, no click, no Quart.

    Use as an async context manager — model + session are created on
    entry and torn down on exit (log_usage + aclose).
    """

    def __init__(
        self,
        config: Config,
        options: CaptionJobOptions,
        *,
        cache: HTTPResponseCache | None = None,
        response_logger: ResponseLogger | None = None,
        http_timeouts: HTTPTTimeouts | None = None,  # connect/read/write/pool seconds
        expected_change_registrar: Callable[[list[str]], None] | None = None,
    ): ...

    async def __aenter__(self) -> CaptioningRunner: ...   # creates APICaptioner + AsyncSession
    async def __aexit__(self, *args) -> None: ...        # model.log_usage() + async_session.aclose()

    async def caption_image(self, image: DatasetImage, callbacks: CaptioningCallbacks) -> str:
        """Stream + accumulate + save. Returns the saved caption text ("" if empty).

        - Calls callbacks.on_image_started, on_token (per token), on_image_captioned.
        - On exception: callbacks.on_image_error, re-raises so callers can decide policy.
        - Before writing caption/TOML/history, calls
          expected_change_registrar([...]) so the API watcher can suppress events.
        - The draft file path is used instead of caption/TOML/history when
          options.draft is set.
        """

    async def caption_image_dry_run(self, image: DatasetImage, callbacks: CaptioningCallbacks) -> str:
        """Stream + accumulate. Does NOT save. Used by CLI's interactive flow where
        the user may retry/reject before committing. Returns the caption text."""
```

**Key design decisions:**

- **Context manager** — model + session lifecycle is naturally scoped to the captioning run. CLI and API both use `async with runner: ...`.
- **Callbacks are passed per-call** (not stored on the runner) — the CLI passes click-emitting callbacks; the API passes SSE-emitting callbacks. Different runs can use different callbacks even with the same runner.
- **`expected_change_registrar` is a constructor-time dependency** — it's an API concern (watcher suppression), but the runner has to know about it because the writes happen inside the runner. The CLI passes `None`; the API passes a closure that calls `dataset_watcher.expect_file_change(name, path)` for each path.
- **`caption_image_dry_run` exists because the CLI's interactive flow** generates captions multiple times per image (continue / retry / reply) and only saves once after the user accepts. The API never needs it. The two methods share an internal `_stream_image()` helper to keep the streaming logic DRY.
- **Cancellation** — the runner doesn't own a `stop_event`; callers pass one in via `options` (a new field `stop_event: asyncio.Event | None = None` on `CaptionJobOptions`), and the runner checks it between images and inside the streaming loop's `CancelledError` handler. The CLI doesn't need it (KeyboardInterrupt is propagated); the API passes its job's `_stop_event`.

## Phases

### Phase 1 — Extract shared core (foundation, no behaviour change)

1. Create `yadc/core/captioning/` package with `__init__.py` re-exports.
2. Move `CaptionJobOptions` from `yadc/api/services/captioning.py` to `yadc/core/captioning/options.py`. Re-export from the API module for backward compat.
3. Move `apply_config_overrides` and `resolve_template` to `yadc/core/captioning/loader.py`. Re-export from the API.
4. Add `load_dataset_config()` to `loader.py` — currently inlined in CLI's `_load_dataset` and API's `preflight_images` + `_ado_run`. Pick the CLI's behaviour as canonical (it has more knobs: `user_config` merge).
5. Run full test suite — everything must still pass with only import-path changes.

**Files touched:** `yadc/core/captioning/{__init__,options,loader}.py` (new), `yadc/api/services/captioning.py` (re-exports + a few imports removed).

### Phase 2 — Create `CaptioningRunner`

1. Implement `CaptioningCallbacks`, `CaptioningStatus`, and `CaptioningRunner` in `yadc/core/captioning/runner.py` per the design above.
2. Add `stop_event: asyncio.Event | None = None` to `CaptionJobOptions` (and to the loader's option-merge path if needed).
3. Add `tests/core/captioning/test_runner.py` with unit tests:
   - `__aenter__` creates the model with the expected kwargs
   - `caption_image` streams tokens and invokes `on_token` for each
   - `caption_image` invokes `on_image_started` before and `on_image_captioned` after
   - `caption_image` invokes `on_image_error` + re-raises on model exception
   - `caption_image` saves caption/TOML/history; uses draft path when `options.draft` is set
   - `caption_image` calls `expected_change_registrar` with the right paths before writes
   - `__aexit__` calls `model.log_usage()` and `async_session.aclose()`
   - `caption_image_dry_run` does not write any files
   - Cancellation: setting `stop_event` between images stops the batch cleanly
4. Add `tests/core/captioning/test_loader.py` for `load_dataset_config`:
   - Loads TOML, parses Config
   - Applies `CaptionJobOptions` overrides (api url/token/model_name, prompt, max_tokens, image_quality, reasoning, rounds)
   - Merges user_config (CLI path)
   - Resolves template via the user/builtin/default chain
   - Resolves images and applies the overwrite/draft filter
   - Raises on missing config file
5. Run full test suite. New tests pass; existing tests untouched.

**Files touched:** `yadc/core/captioning/runner.py` (new), `yadc/core/captioning/options.py` (add `stop_event`), `tests/core/captioning/test_runner.py` (new), `tests/core/captioning/test_loader.py` (new).

### Phase 3 — Refactor `AsyncCaptionJob` to use the runner

The API's `AsyncCaptionJob` is currently a self-contained job class. After this phase, `_ado_run` becomes a thin wrapper that builds a runner and loops over images. The job still owns: job_id, state tracking (`_status`, `_processed`, `_errors`, `_api_url`, `_api_model_name`), SSE event emission, the `_cleanup_async` lifecycle, the `stop_event`, and `preflight_images` (which now calls `load_dataset_config`).

1. Replace `preflight_images` body with a call to `load_dataset_config()`. Keep its signature.
2. In `_ado_run`:
   - Replace the model-creation block with `async with CaptioningRunner(...) as runner:`.
   - Replace the `for img in to_do: await self._acaption_one(...)` block with `await runner.caption_image(img, callbacks)`.
   - `callbacks` emits `CaptioningStatusEvent`, `ImageCaptionStartedEvent`, `ImageCaptionedEvent`, `ImageCaptionErrorEvent` based on the runner's events. State updates (`_processed += 1`, `_errors += 1`, etc.) move into the callback closures.
   - `expected_change_registrar` closure calls `self._dataset_watcher.expect_file_change(self._dataset_name, path)` for each path before writes.
3. Delete `_acaption_one` and the now-unused model-create code.
4. Run API tests. `tests/api/test_captioning.py` and `tests/api/test_captioning_unit.py` should still pass (the API surface is unchanged).

**Files touched:** `yadc/api/services/captioning.py` (substantial rewrite of `_ado_run`, delete `_acaption_one`).

### Phase 4 — Refactor `cli_caption.py` to use the runner (the user's stated goal)

The CLI is the more interesting case because of the interactive flow. Goal: keep every existing interactive feature (retry, edit, prompts, reply, multi-round, draft, overwrite) working unchanged from the user's perspective, but rewire the captioning mechanics to use the shared runner.

1. Replace `_load_dataset()` with a call to `load_dataset_config()` (from the loader). The CLI's `merge_user_config` step moves into the loader. Keep the dataset stream → file path handoff: write the stream to a temp file (or use `dataset.name` if it's a real path) and pass that to the loader. This is needed because the loader takes a path, not a stream — see the open question below.
2. Replace `_predict_caption_one_shot` and `_predict_caption_rounds`:
   - The streaming core (accumulate tokens, call `on_token`, call `model.log_usage` etc.) moves into `runner.caption_image_dry_run` (already done in Phase 2).
   - The CLI wraps each call with `click.echo` output via `callbacks.on_token` and timing logs.
3. In `_caption` (the interactive loop):
   - For "continue" / "retry" / "reply" actions, replace `_predict_caption_one_shot(...)` / `_predict_caption_rounds(...)` with `await runner.caption_image_dry_run(image, callbacks)`.
   - The interactive action menu (quit/skip/continue/retry/edit/prompts/reply/clear replies) stays as-is. The per-image flow becomes: outer action loop → runner streams → user accepts → outer save.
   - The outer save block (`if save_draft: dataset_image.write_draft(...) else: ...update_caption + save_history`) stays in the CLI (it already runs once per image at the end of the action loop).
4. In `_caption_async`:
   - Build a `CaptionJobOptions` from the click kwargs (small mapping function at the top of the file).
   - Replace the cache / response_logger / model-create / async_session blocks with `async with CaptioningRunner(config, options, cache=..., response_logger=...) as runner:`.
5. Run CLI tests. `tests/cli/test_cli_official.py` and `tests/cli/test_cli_local.py` exercise the integration path — they should pass with no changes (same CLI surface). Any unit-level tests of the old internal helpers will need updating.

**Files touched:** `yadc/cli_caption.py` (substantial rewrite of `_load_dataset`, `_predict_caption_*`, `_caption`, `_caption_async`).

### Phase 5 — Cleanup

1. Remove dead code:
   - `_resolve_template` in `cli_caption.py` (replaced by loader + runner)
   - The duplicate HTTP session / model creation blocks
   - The duplicated `apply_config_overrides` / `resolve_template` re-exports in the API (keep only `CaptionJobOptions` re-export for backward compat with existing imports in controllers and tests)
2. Update architecture docs:
   - `docs/backend/cli.md` and `docs/backend/api.md` to mention the new shared package
   - `docs/captioning-workflow.md` to reference `CaptioningRunner` as the canonical loop
   - `docs/cli-cmd-structure.md` — note that `caption` is the one command without a `cmd/caption/` package (it's now `core/captioning/`); document the rationale (shared with API)
3. Add `docs/captioning-runner.md` (new architecture doc) covering:
   - Package layout
   - `CaptioningRunner` lifecycle (context manager)
   - `CaptioningCallbacks` semantics
   - How the CLI and API use it (one short example each)
4. Update the `cli-cmd-structure` doc since `caption` no longer fits the "click command + cmd/ package" pattern — the user can decide whether to keep the doc as-is and add a footnote, or to revise.
5. Run lint + type-check + full test suite.

**Files touched:** `yadc/cli_caption.py` (dead code removal), `yadc/api/services/captioning.py` (re-export cleanup), `docs/backend/{cli,api}.md`, `docs/captioning-workflow.md`, `docs/cli-cmd-structure.md`, `docs/captioning-runner.md` (new).

### Phase 6 — Unify CLI + API logging (deferred)

**Rationale for last phase:** the captioner code (`yadc/captioners/api/utils/*.py`) uses `yadc.core.logging` directly. Once the runner is in place, the surface area of what the unified logger needs to do is fully visible, and we can make interface decisions (Protocol vs ABC, default-factory vs explicit, API factory shape) with the right context.

**Problem today:**
- `yadc/core/logging.py` is the CLI's logger system: custom `_logger` wrapper with a `trace()` level, module-level state, `set_level`/`set_handler` global mutators. ~17 files use `from yadc.core import logging`.
- `yadc/cli_logging.py` is the `ClickHandler` (a `logging.Handler` subclass that routes through `click.secho`).
- `yadc/api/modules/logging_factory.py` is a separate DI-managed `LoggingFactory` that returns raw `logging.Logger` (no `trace()`), takes an int level, and calls `logging.basicConfig` at init.
- When the API runs the captioner, both systems are active at once: `yadc.core.logging` (used by captioner code) and `LoggingFactory` (used by API services). Awkward coexistence with no shared interface.

**Proposed shape:**

```
yadc/core/logging.py
  Logger             # Protocol: trace/debug/info/warning/error/exception/addHandler/setLevel
  PythonLogger       # concrete impl wrapping logging.Logger (adds trace())
  LoggingFactory     # holds default handler + level, creates per-name loggers, mutators
  get_logger(name)   # module-level: uses default factory
  set_level/set_handler  # module-level: delegates to default factory
  TRACE_LEVEL        # constant

yadc/cli_logging.py
  ClickHandler       # unchanged — just a logging.Handler subclass

yadc/api/modules/logging_factory.py
  LoggingFactory(Service)  # thin DI wrapper around core LoggingFactory
```

- **CLI hooks:** `cli.py` does `logging.set_handler(ClickHandler())` + `logging.set_level("INFO")` at startup — same calls as today, but now routed through the unified factory.
- **API hooks:** `yadc/api/modules/logging_factory.py` becomes a thin DI wrapper that builds a `LoggingFactory` from `yadc.core.logging` with an API-shaped `StreamHandler` (structured format from the `logging-format` memory).
- **Captioner code:** unchanged at the call site. `from yadc.core import logging; _logger = logging.get_logger(__name__)` keeps working — now goes through the unified factory, so the API's formatter and level apply automatically.

**Decisions deferred until after Phase 5 lands:**

1. **Interface style.** Protocol (structural, duck-compatible with `logging.Logger`) vs ABC (nominal) vs concrete class only. The user wants to see how the captioning refactor shapes up before deciding — once the runner's logging surface is concrete, the right call should be obvious.
2. **API `LoggingFactory` shape.** Thin DI wrapper (keeps existing constructor signatures in services) vs deletion (services take the core factory directly). Smaller diff vs cleaner long-term.
3. **Module-level default factory.** Preserve the current `from yadc.core import logging; logging.get_logger(__name__)` pattern (default factory lazily created on first call) vs require explicit factory construction (cleaner, bigger diff).
4. **Backward-compat re-exports in `yadc.core.logging`.** Keep the current module-level `get_logger` / `set_level` / `set_handler` (so all ~17 call sites keep working) vs break them and update call sites in this phase.
5. **Where the `ClickHandler` lives.** Stays in `yadc/cli_logging.py` (current location, CLI-only concern) vs moves to `yadc/core/logging/handlers/click.py` (consistent with the new logging package layout). Probably stays put — it's a CLI concern.

**Files touched (TBD):** `yadc/core/logging.py` (refactor), `yadc/api/modules/logging_factory.py` (rewrite as wrapper), `tests/core/test_logging.py` (new). CLI and captioner call sites likely unchanged if decisions #1, #2, #3, #4 favor backward compat.

**Testing:** `tests/core/test_logging.py` for the core factory (handler swap, level change, per-name caching, `trace()` works). Existing CLI/API tests are unaffected.

## Open questions

These need a decision before / during the relevant phase. Listed in order of impact.

1. **`load_dataset_config` input: file path only.** ~~The CLI currently takes `dataset` as a `click.File("r")` so it can read from stdin (`yadc caption -`).~~ **Decision (2026-06-04): drop stdin support.** The loader always takes a path. The CLI's `click.File("r")` arg becomes `click.Path(exists=True, dir_okay=False, readable=True)` (path only). This drops the undocumented `yadc caption -` path; if anyone needs it back later, they can add a temp-file wrapper.

2. **Should the runner expose the underlying `APICaptioner` for advanced CLI flows (multi-round, reply history)?** The CLI's multi-round and reply flows currently build `caption_rounds: list[CaptionerRound]` and `extra_messages: list[ReplyRound]` and pass them to the model. After the refactor, these need to be passed to the runner somehow. **Decision (2026-06-04):** extra kwargs on `caption_image` / `caption_image_dry_run` — `await runner.caption_image_dry_run(image, callbacks, *, caption_rounds=None, extra_messages=None)`. Clean, doesn't bloat `CaptionJobOptions`, and the API just always passes `None`. **Note:** multi-round / reply history is currently CLI-only. The web UI exposes the related fields in `CaptionJobOptions` (`rounds` is there; `extra_messages` / `caption_rounds` are not), but it's unclear whether the API actually supports them end-to-end — out of scope for this refactor, but the runner design is ready for them whenever the API side is.

3. **`CaptioningCallbacks` style: dataclass with `Optional[Callable]` vs. class with default no-op methods.** Dataclass is cheaper to construct (only specify what you need); class is more discoverable. Recommendation: **dataclass** (matches the project's `pydantic` style of explicit-everywhere).

4. **Where does `model.log_usage()` get called?** Currently both the CLI and the API call it at the end of the run. Two options: (a) the runner's `__aexit__` calls it unconditionally; (b) the runner calls it via a `callbacks.on_usage()` and callers do the actual `log_usage()` call. Recommendation: **(a)** — `log_usage` is intrinsic to the model and should happen when the model is torn down. The API's existing behaviour is to call it after the loop, which is equivalent.

5. **Should the runner take a `DatasetImage` list or a `Config` and re-resolve?** Re-resolving would couple the runner to dataset resolution. Recommendation: **the loader returns `(Config, list[DatasetImage])` and the runner takes both** — the caller decides which `base_dir` to use for path resolution.

6. **`CaptionJobOptions.image_ids` is API-specific** (single-image mode). Should the CLI's `CaptionJobOptions` ignore it, or should the loader filter images by it? Recommendation: **the loader filters** — it's a clean place to express "only caption these specific images", and the CLI just always passes `image_ids=None`.

## Risks

- **Interactive flow coupling** — the CLI's `_caption` action menu is tightly integrated with the streaming step. Splitting "stream" and "save" requires careful reordering so that retry / edit / reply still work. Mitigated by writing a focused unit test for the interactive loop after the refactor.
- **`CaptioningOptions` field drift** — fields added by the API (e.g. `image_ids`, `password`) need to be either ignored by the CLI or supported. `password` is already handled by the loader (passed to `cmd_envs.load_env`). `image_ids` will be `None` from the CLI.
- **Stop-event vs. KeyboardInterrupt in the CLI** — the CLI currently relies on `KeyboardInterrupt` propagating up through `_predict_caption_one_shot` to abort. After the refactor, the runner catches `KeyboardInterrupt` and re-raises (or just lets it bubble); the CLI's outer loop handles it. Need to verify the integration tests' Ctrl-C behaviour is preserved (or accept that it's not covered by tests, which is the current state).
- **The `_caption_async` function rebuilds `CaptionJobOptions` from click kwargs** — every new CLI option needs to be mirrored in the options builder. Mitigated by keeping the builder thin (just `CaptionJobOptions(**kwargs)` after dropping CLI-only keys).
- **Backwards compat for `CaptionJobOptions` re-exports** — controllers and tests import it from `yadc.api.services.captioning`. Keep the re-export indefinitely (low cost).

## Testing

- New unit tests: `tests/core/captioning/test_runner.py`, `tests/core/captioning/test_loader.py`. The runner tests use `unittest.mock.MagicMock` for the model and assert on the callback call sequence + side effects (file writes, expected-change-registrar calls).
- Existing API tests (`tests/api/test_captioning.py`, `tests/api/test_captioning_unit.py`) should pass unchanged after Phase 3 — the API surface (`start_job_async` / `stop_job_async` / `get_status_async` / `JobInfo`) is preserved.
- Existing CLI integration tests (`tests/cli/test_cli_official.py`, `tests/cli/test_cli_local.py`) should pass unchanged after Phase 4 — the CLI command surface is preserved.
- Manual smoke test: `yadc caption test_pedro.dataset --no-stream` (existing integration test command) works end-to-end. Stdin is no longer supported.

## Out of scope (deferred)

- Full DI in the CLI (the user said it would be acceptable but not required — skipping).
- Unifying the rest of the CLI/API surface (envs, configs, templates, datasets, export) — these would each need their own plan.
- Replacing the CLI's `_caption` action-menu with a TUI.
- Making the API host the CLI as subprocesses (or vice versa).
- The "clean up captioning server logs" todo (`todo.md` § Clean up captioning server logs) — the runner will use `logger.info(...)` for its own messages, but the existing `print` / `click.echo` output in the CLI callbacks stays as-is. **This is closely related to Phase 6** — once the logger is unified, the remaining CLI output can be routed through it. Phase 6 will subsume this todo.
- **Phase 6 design decisions** (Protocol vs ABC, API factory shape, default factory vs explicit) — deferred until after the captioning runner lands, so the right call can be made with the runner's logging surface visible.

## Decision log (will become history entries as phases land)

- **2026-06-04 — Proposed.** Initial plan created after codebase investigation.
- **2026-06-04 — Approved with decisions.** User confirmed: (1) shared core lives in `yadc/core/captioning/`, (2) drop stdin support for the dataset arg (loader always takes a path), (3) no DI in the CLI, (4) multi-round / reply history plumbing is done via extra kwargs on `caption_image` / `caption_image_dry_run` rather than on `CaptionJobOptions`.
- **2026-06-04 — Phase 6 added (logging unification).** User asked to add a logging refactor as the last phase, with the implementation shape sketched but specific design decisions deferred until after the captioning runner lands. The runner's logging surface will inform the choice of Protocol vs ABC, default-factory vs explicit, and API factory shape.
