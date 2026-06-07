---
name: captioning-workflow
description: End-to-end captioning workflow — config loading, image filtering, the shared captioning loop, and the side that wraps it (CLI's interactive menu / API's SSE event emission).
category: architecture
---

# Captioning Workflow

The end-to-end captioning flow is shared between the CLI (`yadc/cli_caption.py`) and the API (`yadc/api/services/captioning.py`). The shared core is `yadc.core.captioning` (see `captioning-runner` for the runner's interface). Each side adds its own concerns on top.

## High-level shape

```
config.toml
  └─► yadc.core.captioning.load_dataset_config()  ────  Config + list[DatasetImage]
        └─► yadc.core.captioning.CaptioningRunner
              ├─► __aenter__: AsyncSession + APICaptioner + load_model
              ├─► for each image:
              │     └─► caption_image / caption_image_dry_run
              │           ├─► on_image_started, on_token (×N), on_image_captioned / on_image_error
              │           └─► save (or dry-run)
              └─► __aexit__: model.log_usage + async_session.aclose
```

CLI-specific: interactive action menu (quit/skip/continue/retry/edit/prompts/reply/clear-replies), retry/multiple-stream before committing, click-based output, multi-round captioning.

API-specific: per-image SSE event dispatch (`ImageCaptionStartedEvent` / `ImageCaptionedEvent` / `ImageCaptionErrorEvent`), `CaptioningStatusEvent` updates, `DatasetWatcherService.expect_file_change()` registration before writes, job cancellation via `asyncio.Event`.

## 1. Config loading (`yadc.core.captioning.load_dataset_config`)

1. Load TOML from `config_path`
2. Optionally merge with a user config (CLI's `--user-config`; via `cmd_configs.merge_user_config`)
3. Apply `CaptionJobOptions` overrides via `apply_config_overrides` — precedence: opts > env > TOML. `cmd_envs.load_env` is called here (with the optional password for password-mode envs).
4. Resolve the prompt template via `resolve_template` (explicit `prompt_template` → user template → builtin → default; raises if a named template can't be found).
5. `parse_config()` → v1/v2 `Config`
6. `resolve_dataset()` → list of `DatasetImage` (relative paths resolved against `config_path.parent`)
7. **Filter** for overwrite/draft:
   - `image_ids` set → return only those specific images (single-image mode; bypasses overwrite filter)
   - `overwrite=False` and not in `image_ids` mode → skip images with existing caption (or draft if `options.draft` is set)
   - `overwrite=True` → include everything

Returns `(config, images)`. Raises `FileNotFoundError` (missing config) or `ValueError` (parse failure, user-config merge failure, template not found, image_ids resolves to empty).

## 2. Captioning loop (`yadc.core.captioning.CaptioningRunner`)

The runner is an async context manager. See `captioning-runner` for the full interface. Briefly:

- `caption_image(image, callbacks)` — stream + save. Fires `on_image_started` / `on_token` / `on_image_captioned`. On error: `on_image_error` + re-raise.
- `caption_image_dry_run(image, callbacks)` — stream only. No save. The CLI uses this for interactive flows (continue / retry / reply) and calls `runner.save_caption(image, caption)` after the user accepts.
- `save_caption(image, caption)` — write the caption (or draft) to disk, calling `expected_change_registrar` first if set.

`caption_rounds`, `extra_messages`, `prediction_context` are per-call kwargs (not on `CaptionJobOptions`) — see `captioning-runner` for details.

### Per-image work, in order

1. `on_image_started(image)` — caller logs / emits started event
2. Stream tokens from `model.predict_stream(...)` → `on_token(token)` per token
3. Strip + accumulate → caption text
4. If non-empty and not dry-run → call `expected_change_registrar([paths])` then save:
   - **Draft mode** (`options.draft`): write `.{name}.draft~` only
   - **Normal**: write `.txt` (caption), `.toml` (current state with caption), `.history~` (append previous caption first if there was one)
5. `on_image_captioned(image, duration_ms)` on success

### Cancellation

- `asyncio.CancelledError` propagates without firing `on_image_error`. The caller (CLI's outer loop, API's `AsyncCaptionJob._ado_run`) decides how to interpret cancellation.
- The CLI doesn't use a stop event — `KeyboardInterrupt` propagates and the outer loop catches it.
- The API uses `asyncio.Event` checked between images in the job's loop.

## 3. CLI's interactive action menu (above the runner)

The CLI implements the action menu in `cli_caption._caption`. The runner is just the "stream + save" inner step. For each image:

| Key | Action | How it uses the runner |
|-----|--------|------------------------|
| `q` | quit | (no captioning) |
| `s` | skip | (no captioning) |
| `c` | continue | `runner.caption_image_dry_run` (with `extra_messages` if reply history) |
| `r` | retry | `runner.caption_image_dry_run` (regenerate; overwrites prior `caption` local var) |
| `e` | edit | edits the in-memory TOML fields via `$EDITOR`; no model call |
| `p` | prompts | renders the same Jinja2 template via `PromptRenderer` (no model call) |
| `y` | reply | collects a user message; next continue/retry passes it as `extra_messages` |
| `x` | clear replies | clears `reply_history` and `last_prediction_context` |

After the inner action loop, if the user accepted a caption: `runner.save_caption(dataset_image_current, caption)` writes the files.

### Multi-round (`--rounds N`)

When `rounds > 1` and there are no `extra_messages`, the CLI runs N rounds of accept/reject against `runner.caption_image_dry_run(...)` with `caption_rounds=[]` for intermediate rounds, then a final round with `caption_rounds=[all_accepted_rounds]`.

## 4. API's event emission (above the runner)

The API's `AsyncCaptionJob` implements `CaptioningCallbacks` directly and passes `self` to the runner:

| Callback | Job behaviour |
|----------|---------------|
| `on_token` | no-op (SSE clients get the final caption in `ImageCaptionedEvent.caption`) |
| `on_image_started` | dispatch `ImageCaptionStartedEvent` |
| `on_image_captioned` | refresh image index, dispatch `ImageCaptionedEvent`, dispatch `CaptioningStatusEvent` (processed +1) |
| `on_image_error` | log warning, dispatch `ImageCaptionErrorEvent`, dispatch `CaptioningStatusEvent` (errors +1, message recorded) |

The job's `expected_change_registrar` calls `DatasetWatcherService.expect_file_change(dataset_name, path)` for each path before the runner writes, so the inotify-based watcher suppresses the resulting `DatasetChangedEvent`.

`CaptioningService` (in `captioning.py`) wraps the job: starts/stops background `asyncio.Task` instances, dispatches `CaptioningStatusEvent` for job-level transitions (idle → running → done/cancelled/error), and runs a final `DatasetService.rescan_dataset()` after the job completes.

### API Refine endpoint

`POST /datasets/<name>/images/<int:image_id>/refine` accepts `{ feedback, caption? }` plus the standard `CaptionJobOptions` fields. It constructs `extra_messages` as `[ReplyRound(assistant, caption), ReplyRound(user, feedback)]` and passes them to `start_job_async`. The job runs as a single-image overwrite captioning job with the conversation context, producing a new caption that replaces the old one.

## 5. Model setup

The runner's `__aenter__` builds the model:

```python
async with CaptioningRunner(config, options) as runner:
    # runner._model is an APICaptioner (auto-detected backend)
    # runner._async_session is the shared AsyncSession
    ...
```

`__aenter__` builds `headers = {"Authorization": f"Bearer {token}"}` if a token is set, creates `AsyncSession(url, headers=headers, **timeouts)`, calls `APICaptioner.create(url, token, prompt_template, store_conversation, image_quality, reasoning, reasoning_effort, reasoning_exclude_output, cache, response_logger, async_session=...)`, then `await model.load_model(model_name)`.

`__aexit__` calls `model.log_usage()` (prints prompt / response / reasoning token counts) and `await async_session.aclose()`.

## 6. History files

History is only written by `save_caption`, not by `dry_run`. The semantics:

- First-ever caption for an image → no history file exists; `save_caption` writes the new caption to `.txt` and `.toml` but doesn't seed history.
- Subsequent caption (without `--overwrite` of the *history* — there's no flag for this) → `save_caption` first appends the previous caption to `.history~` (`when_not_exists=True`), then writes the new `.txt` and `.toml`, then appends the new state (`when_not_exists=False`).
- Draft mode → no history writes; only the draft file is written.
