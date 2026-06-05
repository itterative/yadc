---
name: captioning-runner
description: Shared captioning runner — the model-create / stream / save loop that both the CLI's `yadc caption` command and the API's `AsyncCaptionJob` build on. Pure async, no DI, callback-driven.
category: architecture
---

# Captioning Runner

The shared captioning loop lives in `yadc/core/captioning/`. Both the CLI and the API use it; neither side reimplements the model-create / stream / save mechanics.

## Package layout

```
yadc/core/captioning/
  __init__.py       # re-exports public API
  options.py        # CaptionJobOptions (Pydantic model)
  loader.py         # apply_config_overrides, resolve_template, load_dataset_config
  runner.py         # CaptioningRunner, CaptioningCallbacks (Protocol), HTTPTTimeouts
```

`core/` is the right home: nothing under `core/` imports from `api/`, `cli_*.py`, or `cmd/`, so there's no circular-import risk.

## Two entry points: loader + runner

```python
from yadc.core.captioning import (
    CaptionJobOptions,        # all knobs for one captioning run
    load_dataset_config,      # load + parse + filter (returns Config, list[DatasetImage])
    CaptioningRunner,         # the stream / save loop
    CaptioningCallbacks,      # Protocol — implement on_self and pass self in
    HTTPTTimeouts,            # connect / read / write / pool seconds
)
```

**Loader** resolves a dataset TOML, applies `CaptionJobOptions` overrides, parses the v1/v2 `Config`, resolves the image list, and applies the overwrite/draft filter. One call replaces the old CLI `_load_dataset()` and API `preflight_images` + re-parse in `_ado_run`.

**Runner** owns the model + session lifecycle and the per-image stream / save loop. Callers iterate over the images themselves and call `caption_image` (save) or `caption_image_dry_run` (stream only) for each.

## Runner lifecycle

The runner is an async context manager. `__aenter__` creates the `AsyncSession` and `APICaptioner`; `__aexit__` calls `model.log_usage()` and `async_session.aclose()`:

```python
async with CaptioningRunner(config, options) as runner:
    for image in images:
        caption = await runner.caption_image(image, callbacks)
```

### `caption_image(image, callbacks, *, caption_rounds=None, extra_messages=None, prediction_context=None)`

Stream + accumulate + save. Calls `callbacks.on_image_started`, `on_token` per streamed token, `on_image_captioned` on success. On a non-cancellation exception, calls `callbacks.on_image_error(image, str(exc), duration_ms)` and re-raises. `CancelledError` re-raises without firing the error callback.

### `caption_image_dry_run(image, callbacks, *, caption_rounds=None, extra_messages=None, prediction_context=None)`

Stream + accumulate. Does not save. Does not call `expected_change_registrar`. Does not fire `on_image_captioned`. Used by the CLI's interactive flow where the user may retry/reject before committing; the caller calls `runner.save_caption(image, caption)` after acceptance.

### `save_caption(image, caption)` — public, sync

Writes the caption (or draft) to disk, calling `expected_change_registrar` (if provided) with the file paths first so the API watcher can suppress inotify events. Honors `options.draft` (writes the draft file instead of caption/TOML/history).

### `model` — property

Returns the underlying `APICaptioner`. Used by the CLI for the `api_type` prefill warning; access via `runner.model.api_type` inside the `async with` block. Asserts if accessed outside.

## `CaptioningCallbacks` Protocol

Four async methods that the runner awaits from its async context:

```python
class CaptioningCallbacks(Protocol):
    async def on_token(self, token: str) -> None: ...
    async def on_image_started(self, image: DatasetImage) -> None: ...
    async def on_image_captioned(self, image: DatasetImage, duration_ms: int) -> None: ...
    async def on_image_error(self, image: DatasetImage, error: str, duration_ms: int) -> None: ...
```

`@runtime_checkable`. Implementations pass `self` to `caption_image` directly. The runner always calls all four methods (no `None` checks); implementations provide no-ops for events they don't care about. `# pyright: ignore[reportUnusedParameter]` is the convention for unused Protocol-required params (keeps the parameter name for Protocol match).

All four methods are `async def` so implementations can do async work directly (e.g. `await asyncio.Lock`) without scheduling it as a separate task. The runner `await`s each method in turn, so callbacks run sequentially in the runner's task.

**`on_image_captioned`** is called only when the caption is non-empty and was successfully saved. **Empty captions** (`""` from the model) → no save, no callback.

**`on_image_error`** is called on any non-cancellation exception, immediately before the exception is re-raised so the caller can decide whether to continue or abort. The runner does not retry on errors.

## How the CLI uses it

The CLI's interactive flow uses `caption_image_dry_run` per image and `save_caption` after the user accepts:

```python
async with CaptioningRunner(config, options, cache=cache, response_logger=response_logger) as runner:
    for image in dataset_to_do:
        # interactive action menu (continue / retry / reply / edit / prompts / skip / quit)
        caption = await runner.caption_image_dry_run(image, cli_callbacks, prediction_context=ctx)
        # user accepts
        runner.save_caption(image, caption)
```

`CLICallbacks` routes `on_token` to `click.echo` (stream mode) or a buffer (no-stream mode); logs warnings in `on_image_error`; the other two are no-ops.

## How the API uses it

The API's `AsyncCaptionJob` implements `CaptioningCallbacks` directly and passes `self` to the runner:

```python
async with CaptioningRunner(
    config,
    options,
    expected_change_registrar=self._expected_change_registrar,  # watcher suppression
    http_timeouts=HTTPTTimeouts(...),                          # from Configuration
) as runner:
    for image in to_do:
        if self._check_stop():
            break
        await runner.caption_image(image, self)  # self implements CaptioningCallbacks
```

The job's `on_token` is a no-op (no SSE token stream), `on_image_started` / `on_image_captioned` / `on_image_error` dispatch `ImageCaptionStartedEvent` / `ImageCaptionedEvent` / `ImageCaptionErrorEvent` and update job state.

## Multi-round and reply history

The CLI supports multi-round captioning (`--rounds N`) and reply history (the "reply" interactive action). Both are passed as kwargs to `caption_image` / `caption_image_dry_run`; they are *not* on `CaptionJobOptions`:

- `caption_rounds: list[CaptionerRound] | None` — accepted intermediate rounds (for the final round's context)
- `extra_messages: list[ReplyRound] | None` — multi-turn conversation (the "reply" action's history)
- `prediction_context: PredictionContext | None` — if provided, the caller can read the populated context (e.g. CLI's reply flow reads `reasoning` for the next assistant turn). If `None`, the runner creates a fresh one and discards it.

The API doesn't use any of these today (single-shot jobs); the runner supports them so the API can opt in later.

## See also

- `captioning-workflow` — end-to-end flow (config loading, filtering, the runner loop, saving) for the CLI and API
- `captioner-architecture` — the `APICaptioner` and per-backend captioners the runner builds
- `dataset-system` — `DatasetImage` model and the caption/draft/history sidecar file conventions
