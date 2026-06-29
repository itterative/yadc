---
name: tagger-architecture
description: ONNX image-tagging subsystem — Tagger abstraction, TaggerResult with categorized output, multiprocessing client/server, and the dataset-image API endpoint.
category: architecture
keep_updated: true
---

# Tagger Subsystem

The tagger subsystem (`yadc/taggers/`) runs ONNX image-classification models
in a separate process and exposes a typed `TaggerResult` with categorized
output. The API endpoint tags images in a dataset by `image_id` (no
client-side image upload).

## Public types

| Type | Location | Purpose |
|------|----------|---------|
| `Tagger` | `yadc/taggers/base.py` | Abstract base — `load_model` / `unload_model` / `predict` |
| `TaggerResult` | `yadc/taggers/base.py` | Dataclass with `tags: dict[str, float]` and `categories: dict[str, list[str]]` |
| `OnnxTagger` | `yadc/taggers/onnx.py` | Concrete ONNX Runtime implementation; downloads from HF Hub when `repo_id` is set |
| `apply_thresholds` | `yadc/taggers/onnx.py` | Pure helper — drops tags below per-category thresholds |
| `replace_underscores` | `yadc/taggers/postprocessing.py` | Pure helper — turns `long_hair`→`long hair` (kaomoji-guarded); no-op returns the same object |
| `TaggerServer` / `TaggerClient` | `server.py` / `client.py` | Multiprocessing boundary (`multiprocessing.Queue`) |
| `TaggingService` | `yadc/api/services/tagging.py` | DI service — owns the subprocess lifecycle |
| `TaggingThresholds` | same | Per-category threshold bundle (rating / general / character) |

## Model source (local path vs HuggingFace Hub)

`OnnxTagger.load_model()` resolves the model from one of two sources:

- **Local path** (existing flow): when `repo_id` is empty, the
  `model_path` argument to `load_model` is used directly. Labels are
  loaded from `label_path` (constructor arg) or auto-discovered
  from `<model_dir>/selected_tags.csv` by `TaggingService`.
- **HuggingFace Hub** (new flow): when `repo_id` is set in the
  constructor kwargs, `load_model` calls
  `huggingface_hub.hf_hub_download(repo_id, repo_model_filename)`
  and `hf_hub_download(repo_id, repo_label_filename)`. The returned
  cached paths are used as the model + label files. The `model_path`
  argument to `load_model` is ignored.

The download happens **in the worker process** (not the main API
process), so the main process doesn't need network access. Downloads
are cached by HuggingFace in `~/.cache/huggingface/hub/` so repeat
starts are fast (just ETag revalidation).

Filenames default to SmilingWolf's convention (`model.onnx`,
`selected_tags.csv`) but are configurable via
`tagger_repo_model_filename` / `tagger_repo_label_filename` for
non-SmilingWolf repos.

## Output shape

`TaggerResult.tags` is the flat `tag → score` map. `TaggerResult.categories`
groups the same tags by name (e.g. `{"rating": [...], "general": [...], "character": [...]}`).
Every tag in a category's list also appears in `tags`. Empty `categories`
means a flat (uncategorized) tagger.

For SmilingWolf / WD models (the `wd-*` family on HuggingFace), the
`selected_tags.csv` label file is parsed by `load_labels` and the
`category` column is mapped:

- `9` → `rating`
- `0` → `general`
- `4` → `character`

A flat `.txt` label file (one tag per line) is also supported for
non-categorized models; detected by extension.

## OnnxTagger preprocessing

The pipeline is parameterized by a `PreprocProfile` (`yadc/taggers/onnx_preprocess.py`)
controlling channel order, normalization, and whether to apply sigmoid
to the model output. Two built-in profiles:

- `WD_TAGGER_PROFILE` (default) — NHWC + BGR + no normalization + no
  sigmoid. Matches SmilingWolf's wd-tagger convention, which bakes
  /255, NHWC→NCHW transpose, and sigmoid into the ONNX graph.

- `TIMM_PROFILE` — NCHW + RGB + ImageNet normalization + sigmoid.
  Standard PyTorch / timm convention; used by e.g.
  `animetimm/convnextv2_huge.dbv4-full` and most classifier exports
  that don't bake preprocessing in.

The pipeline matches the active profile:

1. White-canvas composite for RGBA / palette / LA modes.
2. Fit (preserve aspect ratio) then pad with white to the target size.
   Square targets short-circuit the resize step.
3. Resize to the model's expected input size (auto-resolved from
   `input_meta.shape`; falls back to `profile.default_input_size` for
   symbolic H/W dims).
4. Cast to `float32`.
5. Channel flip if `profile.channel_order == "bgr"`.
6. Normalization if `profile.normalize == "imagenet"`:
   /255 → subtract mean → divide by std with ImageNet statistics.
7. Layout transpose: NCHW hosts get `(H, W, C) -> (C, H, W)`.
   NHWC hosts stay as-is.
8. `np.expand_dims` for the batch axis → batched `float32` tensor.

Layout auto-detection: `resolve_input` parses `input_meta.shape` in
three layers — concrete channel count (1/3/4 at dim 1 or 3), symbolic
dim names (`"num_channels"` / `"channels"` at dim 1 vs dim 3, or
`"height"` / `"width"` at dim 2/3), with concrete values winning over
name hints when they disagree.

Post-inference: if `profile.apply_sigmoid` is true, the raw model
output is converted to probabilities via `1 / (1 + exp(-x))` before
reaching `TaggerResult` (lets `apply_thresholds` work on the same
range wd-tagger produces). `predict()` records
`shape/dtype/min/max/mean` of the preprocessed tensor at DEBUG level
so the host-side pipeline is observable in logs.

Sigmoid is **not** applied: wd-tagger models return post-sigmoid probabilities.

## Multiprocessing protocol

`TaggerServer` spawns a daemon `multiprocessing.Process` named
`yadc-tagger` (via the `forkserver` start method — see the comment in
`server.py` for why not `fork`). The worker loads the model then loops
over the request queue. Wire format (dicts over `multiprocessing.Queue`):

- Startup: worker emits `{"id": 0, "ready": bool, "error": str}` once the model is loaded (`ready=False` on failure).
- Tag: request `{"id": int (>0), "image_bytes": bytes}` → response `{"id": int, "result": TaggerResult, "error": str}`.
- Heartbeat: worker emits `{"id": 0, "heartbeat": True}` every `HEARTBEAT_INTERVAL_SECONDS` (15s) while idle. Shutdown: main process sends `None` on the request queue.

`start()` / `tag()` are **bounded**: they poll the response queue at a
short `poll_interval` (`Queue.get(timeout=...)`, capped at
`POLL_INTERVAL_SECONDS`, default 1s) and check `process.is_alive()` on
every timeout. This poll interval is **decoupled from the worker
heartbeat** — it bounds how fast a dead/killed worker is detected (and
how fast a cancel/kill unwinds), while the heartbeat is just the
worker's idle liveness signal. A dead worker is detected within
~`poll_interval`; a wedged (alive but unresponsive) worker is detected
after `RESPONSE_TIMEOUT_SECONDS` (120s). Control messages (`id == 0`)
are drained — only a response matching the request `id` is returned.
This prevents a crashed/hung worker from blocking the request handler
indefinitely.

`stop()` sends the graceful `None` sentinel and joins cooperatively;
`kill()` skips the sentinel and goes straight to `terminate` →
(short join) → `kill` (SIGKILL), used to interrupt an in-flight
inference (ONNX `session.run` can't be stopped mid-call except by
ending the process).

`TaggerClient` wraps `start` / `tag` / `stop` in `asyncio.to_thread` so
the subprocess is invisible to async callers.

## API endpoints (`yadc/api/controllers/api_tagging.py`)

- `POST /datasets/<name>/images/<int:id>/tag` — **synchronous** single-image tag.
  Used by the interactive Tag button in the image detail UI; the result
  populates the prune grid immediately (no job/SSE round-trip). Optional
  body: per-category threshold overrides. Returns a `TaggerResult` JSON.
  503 when the tagger isn't configured; 404 for a missing image.
- `POST /datasets/<name>/tag` — start a **batch** tagging job
  (`TagJobOptions` body: `image_ids` omitted → whole dataset, `[id]` →
  single image via the job path; threshold overrides; `save`
  `TagSaveOptions`; `source`). Returns `202` + `TagJobInfo`. Progress
  flows via `TagJobStatusEvent` / `ImageTaggedEvent` / `ImageTagErrorEvent`.
- `DELETE /datasets/<name>/tag` — thin delegate over `POST /tagging/cancel`: looks up the dataset's running job_id and cancels it (graceful + escalate). 404 if none running.
- `GET /datasets/<name>/tag` — current job status (`TagJobInfo`).
- `GET /datasets/<name>/tag/jobs` — snapshots of every tracked job in the dataset.
- `POST /datasets/<name>/images/<int:id>/tags` — write **user-pruned**
  tags to a draft / extras without re-running the model (interactive
  save). Body: `{tags, categories, save}`; `save` is `TagSaveOptions`.
- `POST /datasets/<name>/images/<int:id>/tags/preview` — same body as
  save; returns `{"content": "..."}` formatted without touching disk
  (formatter text for `draft`, `[tags]` TOML sub-table for `extras`, empty for `none`). Backs the interactive save bar's live preview;
  the formatters are a server-side registry so they aren't duplicated client-side.
- `POST /tagging/preview` — context-free format preview (no dataset/image).
  Same body as the preview above; used by settings panels to show what a
  save produces from fixed mock data.
- `POST /tagging/cancel` — cancel a batch job and/or interrupt an
  in-flight tag (graceful-first, kill-as-fallback). Body `{job_id?}`;
  `job_id` omitted is the single-image case. Returns `CancelResult`
  (`outcome`: `stopped` / `killed` / `stale_job` / `nothing_running`).

The controller injects `TaggingService`, `DatasetService` (path/image
resolution), and `Configuration` (threshold fallbacks).

## SSE events (`yadc/api/events.py`)

The service dispatches four SSE events via `EventDispatcher`:

- **`TaggerStatusEvent`** (`type="tagger_status"`) — subprocess state transitions.
  - `state`: `"starting" | "ready" | "stopping" | "stopped" | "failed"`
  - `source`: `"hf:<repo_id>"` / `"local:<path>"` / `""` — the
    **server-configured** model (what's actually loaded).
  - `error`: present on `state="failed"`
  - Emitted on every spawn (lazy or respawn), every shutdown (idle
    teardown, app shutdown), and on startup failure.

- **`ImageTagStartedEvent`** (`type="image_tag_started"`) — per-image start
  signal from the batch runner (mirrors `ImageCaptionStartedEvent`).
  `dataset_name`, `job_id`, `image_id`, `file_name`. Drives the tile
  shimmer: the frontend adds the image to `currentlyTagging` on this
  event and removes it on `image_tagged` / `image_tag_error`.

- **`ImageTaggedEvent`** (`type="image_tagged"`) — per-image success.
  - `dataset_name`, `image_id`, `file_name`, `path`, `tags`, `categories`
  - `source`, `duration_ms`
  - Carries the **thresholded** result so other tabs / clients see the
    same view the requester sees.
  - `source` is **frontend-supplied** via the API body (e.g. a model
    name from a dropdown). Falls back to the server-configured model
    when not provided.

- **`ImageTagErrorEvent`** (`type="image_tag_error"`) — per-image failure.
  - `dataset_name`, `image_id`, `error`, `source`, `duration_ms`
  - `source` is the same frontend-supplied / fallback value as above.
  - Not emitted for image-not-found (404 from controller) or
    not-configured (503 from controller) — those are preconditions, not
    tag failures. Other tabs already get the HTTP error response.

- **`TagJobStatusEvent`** (`type="tag_job_status"`) — aggregate progress
  for a batch tagging job (mirrors `CaptioningStatusEvent`). `status`,
  `dataset_name`, `processed`, `total`, `errors`, `job_id`, `error`,
  `source`, `elapsed`. Per-image success/failure still arrive via the
  two events above; this carries the running counts for a progress bar.

The API endpoint accepts an optional `?source=foo` query parameter on
the synchronous single-image `tag` endpoint — the controller reads it
via `request.args.get("source")`. The batch job endpoint takes `source`
in the JSON body (`TagJobOptions`). Both are **frontend-supplied**
labels (e.g. a model name from a dropdown), echoed in the dispatched
events; falling back to the server-configured model when not provided.

The frontend zod schemas + listeners live in `yadc/webui/src/lib/stores/events.ts`
(`tagger_status`, `image_tag_started`, `image_tagged`, `image_tag_error`, `tag_job_status`) and
route into the `lib/stores/tagging/` domain.

## Save path + batch job (`TaggingService`)

All draft/extras writes happen on the backend via `TaggingService.save_tags`:

- **`TagSaveOptions`** (`mode: "none" | "draft" | "extras"`, default `none`):
  - `draft` → `format_draft(draft_format, result)` written via
    `DatasetService.write_draft(name, id, draft_name="tags", text)`.
    Reuses the caption draft sidecar (`.<image>.<name>.draft~`).
  - `extras` → `DatasetService.merge_extras_tags(name, id, tags_dict)`:
    parses the existing per-image TOML with tomlkit, sets the `[tags]`
    sub-table (`general` / `character` lists + a single `rating` string),
    re-serializes preserving other keys, then delegates to
    `update_extras` (which saves history + registers watcher-suppression).
  - `none` → tag only (emit events, write nothing).
  - `overwrite: bool` (default `False`, **batch-only**): when `False`,
    a synchronous preflight (`_filter_skipped_images`, mirrors
    captioning) partitions the resolved image set into `(to_do, skipped)`
    using `_image_already_has_tags` — `draft_name in image_info.draft_names`
    for `draft`, `DatasetService.has_tags_table(name, id)` (parses the
    TOML, checks for a `[tags]` key) for `extras`. Skipped images never
    reach the tagger; `total = len(to_do) + skipped` and `processed` is
    seeded at `skipped` so the bar starts at the right offset. When
    nothing remains (`to_do` empty), `start_tag_job_async` returns
    `TagJobInfo(status="done")` **immediately with no background task**
    — the HTTP response carries `done`, so no SSE event can race it
       back to `running`. The interactive save endpoint always writes regardless.

`format_and_save`-style logic is shared by the batch job (auto-save per
image) and the interactive `POST .../images/<id>/tags` endpoint
(user-pruned tags). Rating is categorical, so extras stores the **top
rating as a string**, not a list.

The **batch job** (`start_tag_job_async` / `stop` / `status` / `list`)
mirrors `CaptioningService`: one job per dataset (`asyncio.Task`),
images resolved up-front (explicit `image_ids` or paginated whole-dataset),
sequential per-image tagging through the shared `tag_image` lifecycle,
`TagJobStatusEvent` emitted on each step. No new subprocess — reuses the
lazy-spawn client. Save failures are per-image best-effort (don't abort
the run); tag failures increment `errors` but the run continues.

## Configuration (`Configuration`)

| Field | Default | Purpose |
|-------|---------|---------|
| `tagger_model_path` | `""` | Local path to ONNX model. The label file is auto-discovered as `<model_dir>/selected_tags.csv` when `tagger_label_path` is empty. Ignored when `tagger_repo_id` is set. |
| `tagger_label_path` | `""` | Explicit local label file. Empty + no repo = auto-discover `<model_dir>/selected_tags.csv`. |
| `tagger_repo_id` | `"SmilingWolf/wd-eva02-large-tagger-v3"` | HuggingFace Hub repo to download the model + labels from. When set, the worker downloads both files from this repo (worker-side, cached by HF). The tagger is enabled when EITHER this OR `tagger_model_path` is set. |
| `tagger_repo_model_filename` | `"model.onnx"` | Filename within the repo for the model. |
| `tagger_repo_label_filename` | `"selected_tags.csv"` | Filename within the repo for the labels. |
| `tagger_preproc_profile` | `"wd-tagger"` | Named preprocessing profile. `"wd-tagger"` (NHWC + BGR, no normalization — SmilingWolf convention that bakes /255 + NHWC→NCHW + sigmoid into the graph) or `"timm"` (NCHW + RGB + ImageNet normalization + sigmoid — standard PyTorch / timm convention used by e.g. animetimm ConvNeXt). Layout (NCHW vs NHWC) is always auto-detected from the model's input shape. Looked up by `yadc.taggers.onnx_preprocess.get_profile`. |
| `tagger_default_input_size` | `0` | Override the profile's default input size when the model has symbolic H/W dims (e.g. `512` for `animetimm/convnextv2`). `0` → use the profile's built-in default (448 for wd-tagger, 512 for timm). |
| `tagger_rating_threshold` | `0.0` | Drop rating tags below this. |
| `tagger_general_threshold` | `0.35` | Drop general tags below this. |
| `tagger_character_threshold` | `0.85` | Drop character tags below this. |
| `tagger_replace_underscores` | `False` | Turn underscored tag names (`long_hair`) into spaces (`long hair`) before the result is dispatched/returned/saved. Kaomojis are always preserved. Off by default to preserve raw model output; opt in per-request from the UI (the request option, also `replace_underscores`, overrides this when set). |
| `tagger_idle_timeout_seconds` | `900.0` | Tear down the subprocess after this many idle seconds. `0` disables teardown (subprocess stays up once started). |
| `tagger_heartbeat_interval_seconds` | `15.0` | Worker pushes a heartbeat while idle at this interval. (Liveness signal only — death detection granularity is `tagger_liveness_poll_seconds`.) |
| `tagger_response_timeout_seconds` | `120.0` | Give up on a wedged-but-alive worker after this many seconds. |
| `tagger_liveness_poll_seconds` | `1.0` | How often the parent polls the response queue while waiting for a tag response / startup. Bounds how fast a dead or killed worker is noticed and how fast a cancel/kill unwinds. Decoupled from the worker heartbeat. |
| `tagger_cancel_grace_seconds` | `1.0` | Grace window before force-killing the subprocess on cancel. After setting a job's stop_event, cancel waits this long for an in-flight inference to finish on its own before terminating the process. |
| `tagger_expected_changes_grace_seconds` | `5.0` | Delay between a batch job ending and clearing its expected-changes source tag. Residual inotify events from the last writes land after the loop exits (kernel buffering + debounce); clearing too early re-tags them with the per-file `source` (`"tagger"`) instead of the job_id, which the originating tab can't suppress. Mirrors captioning's deferred clear. Must exceed `watcher_debounce_seconds` + `watcher_expected_file_ttl`. |

The wd-tagger canonical defaults (0.35 general, 0.85 character) are the
defaults. `rating_threshold=0.0` keeps all ratings (the wd-tagger UI
typically shows the full rating distribution).

## Lifecycle (lazy spawn + idle teardown)

The subprocess is **spawned lazily** on the first request and torn
down after `tagger_idle_timeout_seconds` of inactivity. A periodic
background job (`IDLE_CHECK_INTERVAL_SECONDS = 30.0`, hardcoded in
`yadc/api/services/tagging.py`) checks the idle timer.

- **Startup** (`@event_handler(StartupEvent)`): no eager spawn — just
  schedule the idle check job on `JobScheduler` (if injected).
  `TaggingService` accepts `job_scheduler: JobScheduler | None = None`
  so unit tests can construct it without a scheduler.
- **First request**: `_ensure_running_locked()` builds a `TaggerClient`,
  awaits `client.start()`, and stores it as `self._tagger_client`.
- **Subsequent requests**: the same `TaggerClient` is reused until the
  idle timer fires (or the subprocess dies, in which case the next
  request respawns).
- **Idle check** (`_idle_check_tick`, runs every 30s in a daemon
  thread): if `time.monotonic() - self._last_used_t >= timeout`,
  acquires `self._lifecycle_lock` and calls `_stop_locked_sync()`
  (which blocks the daemon thread on `client._server.stop()` — up to
  ~15s).
- **Per-request**: `tag_dataset_image(...)` →
  `DatasetService.get_image_path(name, image_id)` → `Path.read_bytes()` →
  `TaggerClient.tag(bytes)` → `apply_thresholds(result, ...)` →
  `replace_underscores(result)` (when the option/config is set, post-threshold
  so only surviving tags are touched). The
  lock is held across the `await client.tag(...)` call so the idle
  check can't tear down a server that's actively serving.
- **Shutdown** (`@event_handler(ShutdownEvent)`): acquires the lock
  and awaits `_stop_locked_async()` (the async variant, which calls
  `client.stop()`).

### Concurrency: why `threading.Lock` (not `asyncio.Lock`)

The lifecycle has two callers that need to coordinate:
- Async request handlers (event-loop thread).
- Sync idle check (JobScheduler daemon thread).

`threading.Lock` works for both. **Crucially, async handlers acquire
it via `_acquire_lifecycle()` — a non-blocking acquire polled with
`await asyncio.sleep` — never a blocking `acquire()` on the loop
thread.** A blocking acquire on the loop thread would freeze the loop
and deadlock the unwind of an in-flight tag being cancelled (the loop
must run that coroutine so it can release the lock). The sync idle
check uses non-blocking `acquire(False)` and skips the tick if locked.
Concurrent async requests serialize through the lock (matches the
single-subprocess concurrency limit anyway).

### Cancel (graceful-first, kill-as-fallback)

`TaggingService.cancel_async(job_id=None)` cancels a batch job and/or
interrupts an in-flight tag. ONNX `session.run` can't be interrupted
mid-call except by ending the process, so cancel escalates:

1. **Graceful:** if `job_id` matches a running job, set its
   `stop_event` (the loop breaks at the next image boundary).
   Unknown `job_id` → `stale_job`; terminal/missing → `nothing_running`.
2. **Probe the lifecycle lock** (non-blocking): free → the job is
   between images (or nothing's running) → `stopped`, no kill.
3. **Held** (mid-inference): wait `tagger_cancel_grace_seconds` for it
   to finish on its own; re-probe. Still held → `_force_kill_async()`:
   `await client.kill()` (terminate→SIGKILL), then wait off-loop
   (bounded) for the in-flight `tag()` to surface "worker died"
   (~`poll_interval`) and release the lock, then clear the dead client
   ref under the lock (guarded against clobbering a client respawned in
   the window). Returns `killed`.

`cancel_async` is **synchronous** — it only responds once the kill has
taken effect and the lock is free, so the next request never blocks on
a held lock. With `job_id` omitted (the single-image synchronous-tag
case), there's no job to stop — cancel goes straight to the escalation
check. Captioning is independent (never references the tagger), so
killing the tagger can't affect a running captioning job. Cancel does
**not** stop other datasets' batch jobs (use the per-dataset Stop).

Routes: `POST /api/tagging/cancel` (body `{job_id?}`);
`DELETE /datasets/<name>/tag` is now a thin delegate (looks up the
dataset's job_id → `cancel_async`). Frontend: both the interactive
Tags-tab Cancel button (no job_id) and the batch Stop button (with
job_id) call it; an `isCancellingTagger` store drives a "Cancelling…"
button state during the request.

### Public state

- `is_configured` — `True` iff `tagger_model_path` OR `tagger_repo_id` is non-empty (the service
  *can* serve requests; controller checks this for the 503).
- `is_available` — `True` iff the subprocess is currently running
  (used for diagnostics).
- `is_configured=False` → controller returns 503 `SERVICE_UNAVAILABLE`.
  `is_configured=True` but subprocess not running → controller
  proceeds; the request handler spawns on demand.

## CLI

`yadc tagger <start|tag>` (in `yadc/cli_tagger.py`):

- `start --model PATH [--labels PATH]` — spawn the subprocess and idle.
- `tag --model PATH [--labels PATH] IMAGE` — one-shot tagging; auto-discovers `<model_dir>/selected_tags.csv` if `--labels` omitted.
- Output is grouped by category (rating / general / character) with 4-decimal scores.

## Pure logic

`yadc/cmd/tagger/tag.py` — `tag_image(tagger_cls, model_path, image_bytes, kwargs)` →
spawns a `TaggerServer`, tags, and shuts down. Used by `cli_tagger.py`.

`yadc/taggers/formatters.py` — tag draft formatters (a pluggable registry
keyed by name, like the export backends): `register_draft_formatter` /
`get_draft_formatter` / `format_draft`. Ships `comma` (comma-separated
list, rating excluded — the sd-scripts training caption), `structured`
(category-labeled, for feeding Refine), and `scored` (like `structured`
but each tag carries a 2-decimal confidence, e.g. `1girl (0.95)`, so a
refinement LLM can weigh how much to trust each tag). Category-grouped
formatters emit sections in canonical order: **rating → character →
general** (character leads the subject, general follows as detail);
`comma` uses the same order in its flat list. Tags within each section
are sorted alphabetically so output is stable across re-runs (same
labels + thresholds → byte-identical text) and easy to diff. New text
formats (weighted, JSON) are added by registering a formatter — no
wire-format change. Also `top_rating(result)` (highest-scoring rating
tag) and `extras_tags(result)` (`{general, character, rating}` for the extras
`[tags]` sub-table — rating a single string, categorical not a set; the
`general` / `character` lists are sorted alphabetically so re-runs on the
same labels produce diff-stable extras).

## WebUI surface (`yadc/webui/src/`)

- **Stores** — `lib/stores/tagging/` domain (role-based, mirrors
  `caption/`): `types.ts`, `api.ts` (sync `tagImage`, batch `startTagJob`/
  `stopTagJob`/`fetchTagJobStatus`, interactive `saveImageTags`),
  `status.ts` (per-dataset job map + terminal eviction), `inflight.ts`
  (`currentlyTagging` per-image set), `taggerStatus.ts` (single global
  subprocess-lifecycle slot), `results.ts` (last `TaggerResult` per
  `dataset:image`, fed by both the sync response and the batch SSE),
  `settings.ts` (`tagSettings` storable: thresholds + save options),
  `actions.ts` (`tagSingleImage`, `startBatchTagging`, `stopTagging`,
  `saveImageTagsAction`), `index.ts`. `events.ts` adds the four zod
  schemas + dispatch wiring.
- **Interactive Tags tab** — `lib/components/dataset/detail/Tags.svelte`:
  Tag button (sync) → result grouped by category (rating/general/character)
  as toggle chips with confidence %, pruned by clicking off → save bar
  (mode draft/extras + format/name) → interactive `POST .../images/<id>/tags`.
  4th tab in `ImageDetail`'s `CompactPillTabs`; `onTagsSaved` refreshes
  caption/history so Caption/Extras tabs reflect the write.
- **Batch side panel** — `lib/components/tagging/TagSettingsPanel.svelte`:
  threshold inputs (diff dots vs canonical wd-tagger defaults + per-field
  reset), save-mode/format/name pickers, Start/Stop. New `Tags` tab in
  `DatasetSidePanel`'s `PillTabs` (alongside Caption/Details/Config).
  Thresholds default to **null = server config** (omitted from the request);
  the diff-dot baseline is the canonical defaults (0.0/0.35/0.85) since the
  server's global `Configuration` isn't exposed per-dataset.

## Reference material

`tagger_smilingwolf.py` at the repo root is a sketch of the SmilingWolf
inference flow from another project. It uses `huggingface_hub` +
`pandas` (not in the yadc deps) — `OnnxTagger` re-implements the
preprocessing in stdlib + numpy only. The wd-tagger convention is
faithful; per-category thresholds replace the reference's flat
threshold approach.
