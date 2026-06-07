---
name: batch-captioning-plan
description: Concurrent / batch captioning — multiple in-flight `predict_stream` requests per `CaptioningRunner` job. Builds on the existing async runner; only the per-image loop changes.
status: Done (post-completion follow-ups: ETA fix, multi-active-images UI, id-DESC sort)
category: meta
last_history: 8
---

# Batch Captioning Plan

## Overview

Today, `CaptioningRunner` (and the loops around it in `cli_caption.py` and `AsyncCaptionJob._ado_run`) caption images one at a time — each `await runner.caption_image(img, ...)` call streams a full response before the next request is sent. For hosted APIs with high rate limits (and especially for local vLLM/llama.cpp/ollama backends) this leaves throughput on the table.

This plan adds a `max_concurrent` knob that lets N `predict_stream` requests be in flight at once on a single `APICaptioner` / `AsyncSession`, gated by an `asyncio.Semaphore`. Default is **1** (no behavior change). The work touches:

- `yadc/core/captioning/runner.py` — new `caption_images()` method using `asyncio.gather` + `asyncio.Semaphore`.
- `yadc/core/captioning/options.py` — add `max_concurrent: int = 1`.
- `yadc/captioners/api/openai.py`, `gemini.py` — add a small `asyncio.Lock` around the shared `_api_usage` dict.
- `yadc/cli_caption.py` — `--max-concurrent` flag (Phase 3).
- `yadc/api/services/captioning.py` — pass `options.max_concurrent` into the runner.
- Frontend: `CaptionSettings`, `CaptionOptions`, `CaptionSettingsPanel.svelte`.

## Why this is a runner-level change

The `CaptioningRunner` already:

- Has all per-prediction state scoped locally (predict kwargs, generator locals like `is_thinking`).
- Holds the only shared mutable state (`APICaptioner._api_usage`, an accounting dict) in the inner captioner.
- Wraps `httpx.AsyncClient` via `AsyncSession`, which is concurrency-safe.
- Exposes per-image callbacks (`on_image_started` / `on_token` / `on_image_captioned` / `on_image_error`) that are already `async def`, so a single callback can serve N concurrent invocations without changes.
- Has its only cross-cutting external side effect (`expected_change_registrar`) already protected by a `threading.Lock` inside `DatasetWatcherService`.

`caption_image_dry_run` stays sequential-only — it's for the CLI's interactive flow (user reviews each caption), so parallelism doesn't apply.

---

## Phase 1: Core runner ✅

- [x] `CaptionJobOptions.max_concurrent: int = 1` — job-level only (no TOML field)
- [x] `CaptioningRunner.caption_images(images, callbacks, *, max_concurrent, caption_rounds, extra_messages)` — `asyncio.Semaphore` + `asyncio.gather` + manual sibling cancel on abort
- [x] `BatchAbortedError` raised when `_BATCH_API_ERROR_THRESHOLD` (3) consecutive identical API errors occur
- [x] `_classify_error_message` — distinguishes API-attributable vs image-attributable errors by regex on the normalised message; signature includes the HTTP code so e.g. 401 and 404 don't accumulate
- [x] `BaseAPICaptioner._api_usage_lock: asyncio.Lock` — guards the four `_api_usage` write sites in `openai.py` (two) and `gemini.py` (two). Inherited by all subclasses.
- [x] Tests: `TestCaptionImages` in `tests/core/captioning/test_runner.py` (10 cases) + `tests/captioners/api/test_api_usage_lock.py` (3 cases). All 441 tests pass; ruff + basedpyright clean.

### Implementation notes

- Cancellation: `caption_images` creates tasks eagerly (not lazily), so even with `max_concurrent=3` and 100 images, 100 tasks are scheduled. The semaphore only throttles entry into the model. The cancellation path cancels unfinished tasks and drains via `gather(return_exceptions=True)` before re-raising.
- Error policy: image-attributable errors **reset** the consecutive counter (they break the chain of identical API errors). A different API signature (e.g. switching from 401 to 404) also resets to 1. Successes reset to 0.
- `_api_usage` writes are wrapped in `async with self._api_usage_lock:` — the lock is held only for the dict mutation, so it doesn't serialise predictions.

---

## Phase 2: API ✅

- [x] `AsyncCaptionJob._ado_run` uses `runner.caption_images(to_do, self, max_concurrent=self._opts.max_concurrent)` instead of the per-image loop. Per-image error handling is now done inside the runner; `_ado_run` only re-raises `CancelledError` and `BatchAbortedError` for `_arun` to map to status (`cancelled` / `error`).
- [x] No controller change — `CaptionJobOptions.max_concurrent` is accepted by `POST /datasets/<name>/caption` and `POST /datasets/<name>/images/<id>/caption` for free (extra="ignore"). Field name: **`max_concurrent`** (matches the runner's `Semaphore`-style naming).
- [x] No `JobInfo` change.
- [x] Tests updated: `test_iterates_all_images` → `test_caption_images_called_with_all_images` (verifies `max_concurrent=1` default), `test_caption_images_passes_max_concurrent` (verifies forwarding), `test_continues_after_image_error` deleted (the runner swallows per-image errors now, not the job loop), `test_batch_aborted_error_propagates` added.
- [x] **Frontend `CaptionSettings`**: added `batchSize: number = 1`; `$version` bumped to 2.
- [x] **Frontend `CaptionOptions`**: added `max_concurrent?: number` (the wire field name).
- [x] **Frontend `CaptionSettingsPanel`**: added a "Concurrency" section with a numeric input (`min=1`, `max=32`) and a help tooltip explaining the speed-vs-rate-limit tradeoff.
- [x] **Field-name gotcha**: the wire field is `max_concurrent` (matches the backend `CaptionJobOptions` field). The user-facing label is "Concurrent requests" / "Batch size" in the UI; the store key is `batchSize`. The frontend never sends `batch_size` — that was the original plan wording but it would be silently dropped by `extra="ignore"`. Always `max_concurrent` on the wire.

All 442 tests pass; ruff + basedpyright clean; frontend `svelte-check` 0/0, build succeeds.

## Phase 3: CLI ✅

- [x] `@click.option("--max-concurrent N", type=click.IntRange(min=1), default=1, ...)` added to `cli_caption.caption`.
- [x] `--max-concurrent > 1` with `--interactive` errors out with a clear message ("use --non-interactive") before the runner is created. The error fires from `_caption_async` after the dataset is loaded but before any model setup, so no model is loaded.
- [x] Non-interactive path with `max_concurrent > 1` uses `runner.caption_images(dataset, CLIPrintCallbacks(), max_concurrent=...)` directly. Per-image errors are caught by the runner; the CLI just propagates them through the callbacks.
- [x] Non-interactive path with `max_concurrent == 1` (default) keeps the existing `_caption` action-menu loop. No change.
- [x] Interactive path always uses the legacy `_caption` (sequential, dry-run + save per image). No change.
- [x] `CLIPrintCallbacks` — a new minimal callbacks class for the parallel path. Per-token output is suppressed (would interleave across images); per-image completion and errors are logged with the path + duration.
- [x] Tests: 11 new unit tests in `tests/cli/test_cli_caption.py` covering argument parsing (default, explicit, zero/negative rejected), the interactive+parallel rejection, and the non-interactive parallel path.

**Interactive + parallel deferred** — the user reviewed and confirmed the queue/parallel-review design is non-trivial and out of scope for this pass. The plan keeps the deferred work noted but does not block on it.

### Implementation notes

- The interactive+parallel check runs *before* the runner is created (after `load_dataset_config` returns). No model load is wasted.
- The early check uses `defaults["interactive"]` (the resolved value, with config fallback), not the raw CLI kwarg — so a config with `interactive = true` and CLI `--max-concurrent 4` also errors.
- The legacy `_caption` path is unchanged. It still does the dry-run + save pattern per image with the action menu. Refactoring it to use `caption_images` would be a separate change.
- `CLIPrintCallbacks` reuses the yadc `_logger` for output. In production the `ClickHandler` is installed by `yadc/cli.py`; in unit tests only the default `StreamHandler` is active, so we use `caplog` to inspect log records rather than `result.stderr` (which bypasses CliRunner's capture).

All 453 tests pass; ruff + basedpyright clean.

---

## Phase 4: Tests ✅

All tests were written inline as part of Phases 1–3 (rather than as a separate phase). Total: 32 tests across 4 files.

### `tests/core/captioning/test_runner.py` (`TestCaptionImages` — 10 tests)

- **Sequential path** → `test_max_concurrent_one_processes_serially` — `max_concurrent=1` is equivalent to per-image loop.
- **Bounded concurrency** → `test_max_concurrent_bounds_in_flight_count` — `max_concurrent=3` with 10 images, no more than 3 in flight.
- **Per-image error isolation** → `test_per_image_error_does_not_fail_batch` — one image raising doesn't fail the batch.
- **Cancellation** → `test_cancellation_cancels_siblings_and_propagates` — clean exit, semaphore drained, no deadlock.
- **API error escalation** → `test_api_error_threshold_aborts_batch` — 3 consecutive 401s abort with `BatchAbortedError`.
- **Image errors reset the API counter** → `test_mixed_errors_dont_trigger_abort` and `test_changing_api_signature_resets_consecutive_counter`.
- **Edge cases** → `test_empty_images_is_a_noop`, `test_max_concurrent_must_be_at_least_one`, `test_callbacks_fire_per_image_out_of_order_is_ok`.

### `tests/captioners/api/test_api_usage_lock.py` (3 tests)

- `test_openai_captioner_has_lock` — the lock is created in `BaseAPICaptioner.__init__`.
- `test_concurrent_writes_all_survive` — N concurrent writers don't lose entries.
- `test_lock_is_released_after_use` — a failed writer doesn't deadlock the lock.

### `tests/cli/test_cli_caption.py` (11 tests)

- **Argument parsing** (4) — default, explicit, zero rejected, negative rejected.
- **Interactive+parallel rejected** (2) — `--interactive --max-concurrent 4` errors with clear message + runner never created; `--interactive` with default still uses legacy path.
- **Non-interactive parallel path** (2) — `max_concurrent > 1` uses `caption_images`; `max_concurrent = 1` keeps legacy `_caption`.
- **`CLIPrintCallbacks`** (3) — `on_token` is no-op, `on_image_captioned` logs duration, `on_image_error` logs warning.

### `tests/api/test_captioning_unit.py` (8 tests in `TestAdoRunWithRunner`)

- `test_uses_captioning_runner` — runner built with parsed config + options.
- `test_caption_images_called_with_all_images` — verifies `max_concurrent=1` default.
- `test_caption_images_passes_max_concurrent` — verifies the option is forwarded.
- `test_done_status_on_no_images`, `test_done_status_after_processing`, `test_cancelled_status_when_stop_event_set` — status transitions.
- `test_cancellation_propagates` — `CancelledError` from runner propagates out of `_ado_run`.
- `test_batch_aborted_error_propagates` — `BatchAbortedError` propagates; `_arun` turns it into `status="error"`.

---

## Out of scope

- **Auto-tuning** of `max_concurrent` based on API type / observed rate limits. Could be a follow-up once we have data.
- **Interactive-mode parallel review** (per Phase 3 deferred).
- **Per-image progress reporting** changes — `processed++` counter still increments per image as before. Frontend already matches events by `image_id`, not order, so out-of-order completion doesn't break it.
- **Dynamic concurrency adjustment** based on observed 429s (back off on rate limit, speed back up after). Could be a follow-up.

---

## Open questions / follow-ups

- Should the catastrophic-abort threshold (`3 consecutive API errors`) be configurable per-job? Probably yes — add a `CaptionJobOptions.error_policy` field eventually, but defer the knob for now (just a constant in the runner).
- Should the error mix be reported in `JobInfo`? Useful for the UI to show "3 image errors, 0 API errors" vs "0 image errors, 3 API errors". Defer to follow-up.
- Should `cancel` wait for in-flight requests to drain (with a short timeout) or abandon them? Current `AsyncCaptionJob.request_stop()` cancels the task immediately. Tests should clarify what happens.
