---
date: 2026-06-28
---
# Cancel subsystem + postprocessing + scored formatter

**Context:** After Phases A/B shipped, three gaps surfaced in testing:
single-image and batch tagging had no way to interrupt an in-flight
inference (ONNX `session.run` can run for seconds and isn't
cancellable mid-call except by ending the process); WD-tagger tag names
ship underscored (`long_hair`) which read poorly as captions; and the
draft formatters carried no confidence signal for a refinement LLM to
weigh. This entry records the three enhancements landed together.

**Decision:**

- **Cancel is graceful-first, kill-as-fallback.**
  `TaggingService.cancel_async(job_id=None)` cancels a batch job (set its
  `stop_event` → break at the next image boundary) and/or interrupts an
  in-flight tag. Unknown `job_id` → `stale_job`; terminal/missing →
  `nothing_running`. If the lifecycle lock is held (mid-inference), cancel
  waits `tagger_cancel_grace_seconds` (default 1.0) for it to finish on
  its own, then escalates: `await client.kill()` (terminate → SIGKILL —
  a new `TaggerServer.kill()` that skips the graceful sentinel), then
  waits off-loop (bounded) for the in-flight `tag()` to surface "worker
  died" and release the lock, then clears the dead client ref under the
  lock (guarded against clobbering a client respawned in the window).
  Returns `CancelResult(outcome: stopped|killed|stale_job|nothing_running)`.
  Route: `POST /api/tagging/cancel` (body `{job_id?}`); `DELETE /tag` is
  a thin delegate. Captioning is independent (never references the
  tagger), so killing the tagger can't affect a running captioning job.

- **The cancel endpoint is synchronous** — it responds only once the kill
  took effect and the lock is free, so the next request never blocks on
  a held lock. The frontend leans on this: a local `$state` `isCancelling`
  flag wraps the await (not an SSE-driven store), so the "Cancelling…"
  button window is exactly the request duration and can't desync on
  event timing.

- **Two deadlock/latency fixes came out of testing.** (1) `tag_image` and
  `on_shutdown` acquired `_lifecycle_lock` via a blocking
  `with self._lifecycle_lock:` on the loop thread — that froze the loop
  and deadlocked the killed worker's unwind coroutine (which runs on the
  loop and needs it free to release the lock). Replaced with a
  non-blocking `_acquire_lifecycle()` helper polled with
  `await asyncio.sleep`. (2) Death detection was bound to the 15s worker
  heartbeat, so a kill took ~15s to unwind; decoupled a new
  `poll_interval` (`tagger_liveness_poll_seconds`, default 1.0) from the
  heartbeat — kill unwinds in ~1s.

- **`replace_underscores` is post-threshold postprocessing.**
  `yadc/taggers/postprocessing.py` exposes `replace_underscores(result)`
  mapping `_replace_underscore_for_tag` over `tags` and `categories`
  (kaomojo-guarded via the existing `kaomojis` set; order preserved;
  no-op returns the same object). Applied in `tag_image` **after**
  thresholding so events, the returned result, and saves all carry
  transformed tags, and only surviving tags are touched. Config flag
  `tagger_replace_underscores` (default **False** — preserve raw model
  output) with per-request override (`replace_underscores` on
  `tag_image`, `TagJobOptions`, `TagImageBody`); explicit bool overrides
  config, mirroring threshold semantics. Frontend: a `Checkbox` in
  `TagSettingsPanel`'s Thresholds section, persisted in `tagSettings`
  (`$version` bumped 1→2 with a migrate).

- **`scored` is a confidence-aware draft formatter** extending
  `structured`: same category-grouped shape but each tag carries a
  2-decimal confidence (`1girl (0.95)`), rating included. Lets a
  caption-refinement LLM weigh how much to trust each tag. Registered
  under `"scored"`.

- **Canonical category order is rating → character → general** across
  all category-grouped formatters (`structured`, `scored`) and the flat
  `comma` list (via `_ordered_tags`), defined as `_CATEGORY_ORDER`.
  Character leads the subject, general follows as descriptive detail —
  independent of the input dict order, which the CSV determined.

**Rationale:**

- Graceful-first avoids killing the subprocess when a job is merely
  between images (free lock) or will finish inside the grace window;
  kill is reserved for genuinely mid-inference cases. Synchronous cancel
  keeps the API contract honest (the request resolves to the real
  outcome) and removes a whole class of frontend race.
- Underscores default-off preserves raw model output (round-trippable,
  matches the reference tagger UI); opt-in per-request from the UI when
  the user wants caption-ready text.
- Canonical order removes CSV-order coupling from the formatter output,
  so reordering the label file never silently changes captions.

**Files touched:** `yadc/taggers/{server.py,client.py,postprocessing.py,
formatters.py,__init__.py}`, `yadc/api/{configuration.py,services/tagging.py,
controllers/api_tagging.py}`, `yadc/webui/src/lib/{stores/tagging/*,
components/tagging/TagSettingsPanel.svelte,components/dataset/detail/Tags.svelte}`,
`tests/taggers/{test_server.py,test_service.py,test_postprocessing.py,
test_formatters.py}`.

**Plan status:** Phase C added; Phase A/B marked complete in the plan body.
