---
date: 2026-06-04
---
# Phase 2: Runner design deviations

**Context:** The plan described the `CaptioningRunner` design in some detail. Several decisions needed refinement when the actual code was written.

**Decision:** Four small deviations from the plan's design, each documented below.

## 1. Drop `on_status` and `on_usage` callbacks (and `CaptioningStatus`)

The plan included `on_status: Callable[[CaptioningStatus], None] | None` and `on_usage: Callable[[], None] | None` in `CaptioningCallbacks`, plus a `CaptioningStatus` dataclass. These are dropped.

**Why:** The runner has no `run()` batch method — the caller loops and calls `caption_image` per image. The runner therefore has no batch state (processed, total, errors) to report. `__aexit__` calls `model.log_usage()` directly (no callback needed). The caller is responsible for emitting its own status events (e.g. the API's `AsyncCaptionJob` will dispatch `CaptioningStatusEvent` from its own loop). This keeps the runner simple and aligned with the per-call callback model.

## 1a. `CaptioningCallbacks` is a `Protocol`, not a dataclass

`CaptioningCallbacks` was a `@dataclass` with `Optional[Callable]` fields and a None check at every call site. It's now a `@runtime_checkable Protocol` with method declarations.

**Why:** The API's `AsyncCaptionJob` (Phase 3) will pass `self` directly as the callbacks argument — structural typing is a much better fit than building a wrapper object on every call. The runner now calls the four methods directly (no `if callbacks.on_X` check); implementations are responsible for being no-ops when they don't care. Tests use a small `_TestCallbacks` adapter class (also satisfies the Protocol) that captures each event into a list.

## 2. Drop `stop_event` parameter; caller checks it in its own loop

The plan said to add `stop_event: asyncio.Event | None = None` to `CaptionJobOptions` and have the runner check it. The plan also said to add the field to the loader's option-merge path.

**Why:** `CaptionJobOptions` is a Pydantic model; `asyncio.Event` doesn't fit cleanly. More importantly, the runner doesn't need to know about it — the caller's loop checks the stop event between `caption_image` calls (e.g. `if self._check_stop(): break`). The runner's `CancelledError` handling is just re-raise, regardless of stop_event semantics; the caller decides what cancellation means in its context.

## 3. Add `prediction_context` kwarg to `caption_image` / `caption_image_dry_run`

The plan mentioned "extra kwargs on `caption_image` / `caption_image_dry_run`" for multi-round and reply history (`caption_rounds`, `extra_messages`), but didn't include `prediction_context`.

**Why:** The CLI's "reply" flow reads `last_prediction_context.reasoning` / `.reasoning_encrypted` to build a `ReplyRound` for the next turn. The CLI needs to keep a reference to the `PredictionContext` the model populated. The runner accepts a caller-provided `PredictionContext` (or creates one if `None`); the caller can capture the provided one for later use. The API doesn't pass one (the runner creates a fresh one per call and the API discards it).

## 4. Runner always uses `predict_stream()`, not `predict()`

The CLI's `--stream/--no-stream` option currently switches between `model.predict()` (non-streaming HTTP call) and `model.predict_stream()` (streaming). The runner always uses `predict_stream()`.

**Why:** Streaming is a superset of non-streaming — buffering the tokens reproduces the non-streaming UX. The CLI's `--no-stream` mode becomes a UX-level buffer in the `on_token` callback (don't print, just accumulate). This is a small UX change but unifies the runner's code path.

**Files touched:** `yadc/core/captioning/runner.py` (new), `yadc/core/captioning/__init__.py` (re-exports), `tests/core/captioning/test_runner.py` (new, 33 tests), `.pi/agent/memory/plans/cli-api-captioning-shared-runner-plan.md` (status, phases_complete, last_history).
