---
date: 2026-06-29
---
# Initial decisions for the tagger model-swap feature

Lock-in pass before implementation. The plan body stayed mostly
as-stashed; this entry records the eight decisions, the deferred
question, and the implementation order that came out of the pass.

## Decisions

1. **Picker scope** — curated HF repos + a "Local file…" option,
   with the local option carrying a warning ("local models may not
   be supported; ONNX with shape `[N,C,H,W]`, sigmoid-on-output
   models work best — wd-tagger profile is the most-tested path").
   Resolves the regression concern: users on local-paths (animetimm
   et al.) keep their workflow and don't need to push the model to
   HF to use the picker.
2. **SettingsService first consumer** — `tagger.active_model` is the
   first real key in the existing JSON-typed KV store. Other future
   keys (captioning defaults, UI prefs) ride the same infrastructure.
3. **UI location** — `SettingsDialog`, in the **general** tab (deliberately
   lightweight tab, took the swap section rather than growing it
   elsewhere). `TagSettingsPanel` doesn't gain a second picker.
4. **`source` field on `TaggerStatusEvent`** — `_source_label()` no
   longer reads `Configuration.tagger_*` directly; it reads from the
   persisted `self._active_tagger`. `swap_active_model` dispatches a
   fresh `TaggerStatusEvent(ready, source=new_label)` after a
   successful respawn, so other browser tabs refresh without polling.
   The per-image `ImageTaggedEvent.source` (frontend-supplied) is
   unchanged.
5. **Drain timeout** — bounded by `tagger_response_timeout_seconds`
   (120s default). UI shows a busy spinner for the duration.
   Acknowledged high; can be tuned later when manual testing reveals
   a real-world long-inference case.
6. **Concurrent swap requests** — second one is refused with HTTP
   **429** carrying `{"error": "...", "detail": "...", "retry_after_s": N}`.
   Implemented via a second flag (`_swap_in_progress`) so request
   handlers don't have to hold `_lifecycle_lock` for the whole swap.
7. **409 during batch** — `409 {"error": "...", "detail": "a batch
   tagging job is running; stop it before swapping the model"}`. UI
   points at the stop button.
8. **Configuration** — legacy tagger-model flat fields stay in
   `configuration.py` untouched (the local tmp commit on animetimm
   keeps applying). They serve as the **fallback / defaults** at
   first-boot when `SettingsService` has no value. New config knobs
   introduced by the swap feature (e.g. swap-specific timeouts) CAN
   be added as new fields on `Configuration`.

## Deferred

- **`preproc_profile` / `default_size` — independent or rides with
  the active model?** Provisionally **rides with the model**: the
  profile choice is model-specific (animetimm needs `timm`, wd-tagger
  needs `wd-tagger`), so the active selection carries both. To confirm.

## Implementation order

1. **SettingsService wiring** — DI check, hydrate `active_tagger` at
   `TaggingService.__init__`, round-trip test.
2. **`ActiveTagger` schema + `_source_label` rewrite** — new dataclass,
   refactor `_ensure_running_locked` to read from `self._active_tagger`
   (with fallback), update cache key derivation, tests for both
   paths.
3. **Swap infrastructure** — `swap_active_model` with the 429 / 409
   branches; new exception types; `_swap_in_progress` flag; event
   dispatch with the new source.
4. **Endpoints** — `GET /api/tagger/active`, `POST /api/tagger/swap`,
   `GET /api/tagger/models` (static catalog in a new
   `yadc/api/modules/tagger_catalog.py`).
5. **Frontend** — `SettingsDialog` general-tab section (store +
   picker + swap button + toast wiring for 409/429), `is_available`
   indicator derived from `$taggerStatus`.

**Files (planned):** `yadc/api/services/tagging.py`,
`yadc/api/services/settings.py` (no changes — first consumer), new
`yadc/api/modules/tagger_catalog.py`, `yadc/api/controllers/api_tagging.py`,
`yadc/webui/src/lib/stores/tagging/api.ts`, new picker section in
`yadc/webui/src/lib/components/settings/SettingsDialog.svelte`,
`tests/taggers/test_service.py`, `tests/taggers/test_api_tagging.py`
(or similar), `tests/api/test_tagger_catalog.py`,
`docs/tagger-architecture.md` (active_tagger + source-label
section). `Configuration` gain only the new swap-specific knobs.
