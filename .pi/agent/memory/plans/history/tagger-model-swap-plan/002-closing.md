---
date: 2026-06-29
---
# Closing — tagger model-swap delivered

All four phases shipped. Plan is **Complete** as of this entry; will be
moved to `archive/` separately.

## Delivered

**Phase 1 — Backend swap infrastructure**
- `ActiveTagger` Pydantic model (`yadc/api/modules/tagger_catalog.py`) — frozen, kind discriminator, validators for `repo_id` / `model_path` non-empty, `source_label` property matching the SSE format
- `TaggingService._active_tagger` + `_hydrate_active_tagger()` constructor wiring + `set_active_tagger()` / `swap_active_model()` methods + `effective_active_tagger` (synthesizes from flat `Configuration` when no persisted swap)
- `_subprocess_spawn_args()` helper unifies active_tagger + flat-fallback paths (`_subprocess_spawn_args` is the single source of truth at the spawn call site)
- `TaggerBusyError` (→ 409) / `TaggerSwapInProgressError` (→ 429 with `retry_after_s`) typed exceptions
- `_swap_in_progress_lock` (`threading.Lock`) as the atomic concurrent-swap gate, distinct from `_lifecycle_lock`

**Phase 2 — Backend endpoints**
- `GET /api/tagger/active` — `{active, is_available}` reads `effective_active_tagger` so legacy Configuration-only setups (animetimm et al.) reflect the right model without a no-op swap
- `POST /api/tagger/swap` — full `ActiveTagger` body, 409 / 429 / 400 mapping; emits `starting` → `ready` SSE events for cross-tab refresh
- `GET /api/tagger/models` — curated catalog + `profiles: ["wd-tagger", "timm"]` for the Type dropdown

**Phase 3 — Persistence wiring**
- `tagger.active_model` key in `SettingsService` (first real consumer of the previously-unused KV store)
- Hydrate at startup
- Persist AFTER successful respawn; failure logs warning, doesn't roll back

**Phase 4 — Frontend picker**
- `fetchActiveTagger()` / `listTaggerModels()` / `swapTaggerModel()` in `lib/stores/tagging/api.ts`; `swapTaggerModel` returns a discriminated `SwapTaggerResult` (`ok` / `busy` / `in_progress` / `error`) so the action layer never re-parses
- `swapActiveModelAction` in `lib/stores/tagging/actions.ts` dispatches the right toast for each variant
- SettingsDialog general-tab section: "Active: <model> · type <profile>" indicator, dropdown showing repo_id (matching the indicator), description helper text, "Swap" inline next to the dropdown, `.alert-warning` + disabled dropdown/Swap when a batch is running
- Type + Input size controls (hidden for curated HF rows where the catalog row carries defaults; editable for Local)

## Deviations from plan (intentional)

1. **Same-identity no-op went in the backend** (originally deferred to frontend via `canSwap`). The frontend's disabled button prevents the user from triggering the no-op, but the backend short-circuits as well for direct API calls and future clients. Compares full `ActiveTagger` equality (frozen=True → value equality); same repo + different profile is a real respawn.
2. **`currentModel` parsed from `taggerStatus.source`** — dropped. The indicator reads `GET /api/tagger/active` + `effective_active_tagger` directly. The SSE source label is still emitted (for any external listener) but parsing it client-side is unnecessary because the structured endpoint is authoritative.
3. **`effective_active_tagger` synthesis from flat Configuration** — added (not in plan). Lets legacy setups see their model in the picker without a no-op swap.

## Deferred (not blocking)

- **Cross-tab settings dialog sync** — second tab with the dialog open stays stale until reopened. Reusing `TaggerStatusEvent` (state=ready) → re-fetch `GET /api/tagger/active` is the minimal fix; deferred to a follow-up because it's an edge case (multi-tab settings dialog).
- **Custom HF repo_id entry in picker** — deferred per Decision 1.
- **Structured `TaggerStatusEvent.model` field** — superseded by `effective_active_tagger` synthesis.

## Tests

- 1098 backend tests pass (added across `test_active_tagger.py`, `test_active_tagger_persistence.py`, `test_active_tagger_integration.py`, `test_swap_active_model.py`, `test_tagger_swap_endpoints.py`, `test_service.py` for `effective_active_tagger`)
- 64 frontend tests pass
- `swap_active_model` coverage: happy path, busy refusal, concurrent-swap refusal, rollback on respawn failure, persist failure leaves in-memory state, no-op on same selection, no-op skipped when active is None (migration), profile change triggers real swap
- `GET /api/tagger/active` coverage: null when unconfigured, persisted selection + source label, synthesized from legacy Configuration, is_available reflects subprocess state

## Files touched (cumulative across all phases)

Backend:
- `yadc/api/modules/tagger_catalog.py` (new — `ActiveTagger`, `TaggerModelSummary`, catalog constants)
- `yadc/api/services/tagging.py` (active_tagger state, `_hydrate_active_tagger`, `_subprocess_spawn_args`, `swap_active_model`, `effective_active_tagger`, exception types)
- `yadc/api/services/settings.py` (no changes — first consumer of existing KV)
- `yadc/api/controllers/api_tagging.py` (three new endpoints + 409/429 mapping)
- `yadc/api/controllers/utils_json.py` (`TOO_MANY_REQUESTS` enum value)

Frontend:
- `yadc/webui/src/lib/stores/tagging/types.ts` (catalog/selection/swap types)
- `yadc/webui/src/lib/stores/tagging/api.ts` (fetchers + `SwapTaggerResult` discriminated union)
- `yadc/webui/src/lib/stores/tagging/actions.ts` (`swapActiveModelAction`)
- `yadc/webui/src/lib/components/settings/GeneralSettings.svelte` (new picker section)

Tests:
- `tests/taggers/test_active_tagger.py`, `test_active_tagger_persistence.py`, `test_active_tagger_integration.py`, `test_swap_active_model.py`, `test_service.py` (effective_active_tagger)
- `tests/api/test_tagger_swap_endpoints.py`