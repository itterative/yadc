---
name: tagger-plan
description: ONNX image-tagging feature for yadc — base abstraction, ONNX implementation, multiprocessing subprocess, dataset-image API endpoint, CLI, batch tagging job, WebUI surface, cancel, postprocessing.
last_history: 7
---

# Tagger Plan

The tagger subsystem (`yadc/taggers/`) provides ONNX-based image tagging
(primarily the SmilingWolf / WD family of models) with a typed,
categorized output (rating / general / character).

Reference: `.pi/agent/memory/docs/tagger-architecture.md`.

## Status

Core complete and verified end-to-end. The list under **Pending**
tracks follow-up refactors and polish identified after the launch
(user-configurable model, threshold sliders in the interactive Tags
tab, CSV import for tags).

Design history lives in `history/tagger-plan/`:
`001-*.md` (webui/batch phase), `002-*.md` (post-launch enhancements),
`003-*.md` (plan-body prune).

## Completed

1. **Scaffolding + categorized output** — `Tagger` ABC; `OnnxTagger`
   faithful to the wd-tagger convention; `TaggerResult` dataclass
   (`tags` + `categories`); SmilingWolf `selected_tags.csv` + flat `.txt`
   label support.
2. **Multiprocessing client/server** — `TaggerServer` (subprocess owner)
   + `TaggerClient` (async wrapper). Dict-over-`Queue` protocol with
   id-keyed correlation.
3. **Configuration** — model/label paths, HF repo options, per-category
   thresholds, idle teardown, liveness knobs (see the config table in
   `tagger-architecture.md`).
4. **API service + single-image endpoint** — `TaggingService` (DI) with
   startup/shutdown handlers; `POST .../images/<id>/tag` with threshold
   overrides (503 unconfigured, 404 missing image).
5. **CLI** — `yadc tagger start` (idling) / `yadc tagger tag` (one-shot),
   auto-discovering `selected_tags.csv`.
6. **Lifecycle (lazy spawn + idle teardown)** — first request spawns;
   periodic job tears down after `tagger_idle_timeout_seconds`. See the
   architecture doc for the lifecycle + concurrency rationale.
7. **HuggingFace Hub download** — worker downloads model + labels from a
   repo on startup (main server needs no network); HF cache makes repeat
   starts fast.
8. **SSE events** — `TaggerStatusEvent`, `ImageTaggedEvent`,
   `ImageTagErrorEvent` via the `EventDispatcher`.
9. **Phase A — Batch tagging job + backend writes.** Job-based run
   mirroring `CaptioningService` (`POST/DELETE/GET /tag`, `GET /tag/jobs`);
   `TagSaveOptions` (none/draft/extras) with pluggable `TagDraftFormatter`
   registry; extras written as a merge preserving other keys; interactive
   save endpoint (`POST .../images/<id>/tags`). See `history/tagger-plan/001`.
10. **Phase B — WebUI surface.** Interactive Tags tab in `ImageDetail`
    (Tag → prune grid → save bar); batch `TagSettingsPanel` in the dataset
    side panel; `lib/stores/tagging/` domain mirroring `caption/`; SSE
    zod schemas + listener. See `history/tagger-plan/001`.
11. **Phase C — Cancel + postprocessing + scored formatter.**
    `POST /api/tagging/cancel` (graceful-first, kill-as-fallback);
    `replace_underscores` post-threshold (config flag + per-request
    override + UI checkbox); `scored` draft formatter (structured +
    confidence) with canonical character-before-general order. See
    `history/tagger-plan/002`.
12. **Phase D — Batch save robustness.** `overwrite` option (default
    off) skips images that already carry the target artifact via a
    synchronous preflight filter (mirrors captioning) — when nothing
    remains, `start_tag_job_async` returns `done` immediately with no
    background task (HTTP response carries the terminal state, no SSE
    race); progress bar seeded at the skip offset; topbar progress bar
    with serial ETA for batch tagging jobs; deferred expected-changes
    clear (`tagger_expected_changes_grace_seconds`) to stop the
    spurious post-job "files changed" toast; topbar progress bar with
    serial ETA for batch tagging jobs.
13. **Phase E — Preprocessing profiles + sigmoid (animetimm ConvNeXt).**
    New `yadc/taggers/onnx_preprocess.py` module with `PreprocProfile`
    controlling channel order, normalization, default input size, and
    output sigmoid. Two built-in profiles (`wd-tagger`, `timm`); layout
    auto-detection through concrete dims + symbolic dim names
    (`num_channels` / `channels` / `height` / `width`); configuration +
    CLI plumbing; diagnostic logs at INFO (contract) and DEBUG (tensor
    stats). See `history/tagger-plan/004`.
14. **Phase F — Alphabetical sort in draft / extras + confidence
    descending on the frontend prune grid.** Tags within each section
    are sorted alphabetically (plus the uncategorized-fallback
    branch) — `comma`, `structured`, `scored`, and `extras_tags` all
    use the same ordering, so re-runs with the same labels produce
    byte-identical saved output. The interactive prune grid in the
    Tags tab sorts the other way (confidence descending, ties
    alphabetical) since the user is choosing which tags to keep —
    most-likely tags surface first. See `history/tagger-plan/005`.
15. **Phase G — Read-through on the result LRU + bucketed cache
    key.** `tag_image` consults `_tag_results` before spawning the
    subprocess; a hit returns the cached value (re-applied at the
    request's effective thresholds + `replace_underscores` post-hoc)
    with `ImageTaggedEvent` dispatched and skips the byte read /
    lifecycle work. Covers POST `/tag` and the batch job's per-image
    loop under matching bucketed thresholds. The cache key coarsens
    thresholds (`rating=0` always; `general` / `character` floored to
    the nearest `0.2`), so nearby threshold values share a slot.
    See `history/tagger-plan/006`.
16. **Phase H — Memory-aware cache budget.** Replaced the
    count-limited `LRU(500)` with `MemoryLRU(max_bytes=128 MiB,
    size_fn=tamer_result_size)` so large-vocab taggers
    (animetimm ConvNeXt ~12k tags / result) don't dominate the
    working set. New `MemoryLRU` class shares the OrderedDict
    backbone with the existing count-limited `LRU`.
    See `history/tagger-plan/007`.

## Pending (future iterations)

**Backend**

- **User-configurable tagger model.** Let the user pick / swap the model
  from the UI instead of only via `Configuration`. Needs a fair bit of
  refactoring (the model identity is wired through startup) plus new
  endpoints (list available models, get/set the active one, probably a
  reload / respawn of the subprocess).

**Frontend**

- **Threshold sliders in the interactive Tags tab** (currently only in
  the batch `TagSettingsPanel`).

**Open**

- **CSV import for tags** — the export system emits `tags = {rating = [...],
  general = [...], character = [...]}` from the captioning TOML extras.
  Whether the tagger round-trips into that shape via a CSV importer is open.

## Notes / Decisions

High-level rationale (why subprocess, why lazy spawn + idle teardown, why
per-category thresholds, why CPU-only `onnxruntime` default + `[gpu]` extra,
the graceful-first-then-kill escalation, underscores default-off) lives in
`tagger-architecture.md` (current-state) and the history entries below
(decision-time context). The plan body is kept to status + phase pointers.
