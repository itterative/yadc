---
name: tagger-model-swap-plan
description: User-configurable tagger model — let the user pick / swap the active model from the WebUI, drain in-flight work, persist the choice, and surface "currently running" + "available models" to the UI.
last_history: 2
status: Complete
---

# Tagger Model Swap Plan

Lets the user change the active tagger model from the WebUI
(instead of only via `Configuration`). The choice persists across
restarts. Pending in-flight tagging requests are drained before the
subprocess is swapped (the user's already-running call still gets
its result), and swaps are refused while a batch job is running.

Design history lives in `history/tagger-model-swap-plan/` (created on
deviations).

## Decisions (agreed)

| Question | Decision |
|----------|----------|
| Pending-request semantics | **Drain** — wait for in-flight single-image call to release the lifecycle lock (bounded by `tagger_response_timeout_seconds`), then tear down + respawn. Refuse swap (HTTP 409) if any batch job is running. |
| Available-models source | **SmilingWolf HF repos only** for v1 (static curated list). Custom repo_id input deferred. Local file picker deferred. |
| Persistence | **SQLite via `SettingsService`** — first real consumer of the existing (currently unused) KV store. `Configuration` carries the in-memory active selection. |
| Config awareness | Active selection is a field on `Configuration` (not a separate store) — `TaggingService` reads from it like any other knob. |
| Endpoint shape | `GET /api/tagger/active`, `POST /api/tagger/swap`, `GET /api/tagger/models`. |
| Cache | No changes — `TaggerResultKey.model_id` already isolates results per model. Old entries age out via LRU. |
| Cancel semantics | Reuse existing `cancel_async()` if the swap endpoint needs to abort mid-swap (e.g. user clicks Cancel in the UI). |

## Phases

### Phase 1 — Backend swap infrastructure

- Add `active_tagger: ActiveTagger | None` to `Configuration`
  (`ActiveTagger` = small dataclass with `kind: "hf"|"local"`,
  `repo_id`, `repo_model_filename`, `repo_label_filename`,
  `model_path`, `label_path`). `None` falls back to the existing
  flat fields (`tagger_repo_id`, `tagger_model_path`, …) for
  backward compatibility.
- Refactor `_ensure_running_locked` to build the subprocess kwargs
  from `Configuration.active_tagger` first, falling back to the
  flat fields. Single source of truth at the call site.
- Add `TaggingService.swap_active_model(selection: ActiveTagger)`
  — drain → tear down → respawn → persist → emit events.
  - No-op when the running subprocess has the same identity
    (avoids needless churn when the UI re-sends the current pick).
  - Raise `TaggerBusyError` (→ 409) if `any(self.is_tagging(d) for d in ...)`.
  - Persists via `SettingsService` (Phase 3) after successful respawn.
- Tests: drain semantics, swap-during-batch raises, no-op on same
  identity, identity-from-config vs fallback fields.

### Phase 2 — Backend endpoints

- `GET /api/tagger/active` → `{ kind, repo_id, model_path, label_path, is_available }`
  - `is_available` reflects whether the subprocess is currently up
    with the active selection (may be `false` between teardown and
    next request).
- `POST /api/tagger/swap` body: `{ kind, repo_id?, repo_model_filename?, repo_label_filename?, model_path?, label_path? }`
  → returns the resulting state synchronously (200) or 409 if batch
  jobs are running. Pydantic model for the body.
- `GET /api/tagger/models` → static curated list of SmilingWolf HF
  repos with display name + parameter hints (e.g.
  `{"id": "SmilingWolf/wd-eva02-large-tagger-v3", "display": "WD EVA02 Large v3", "params": "..."}`)
- Tests: Pydantic validation (unknown kind, missing fields), active
  returns currently-running model, models list shape.

### Phase 3 — Persistence wiring

- Key in `SettingsService`: `tagger.active_model` with JSON value
  matching the swap body shape (`{kind, repo_id, ..., model_path, ...}`).
- Hydrate `Configuration.active_tagger` at webui startup:
  `SettingsService.get("tagger.active_model")` → `ActiveTagger` or
  `None` (which triggers the flat-fields fallback).
- `swap_active_model` calls `SettingsService.set("tagger.active_model", ...)`
  after the subprocess respawn succeeds. Failure to persist logs a
  warning; the in-memory change still applies for the session.
- Tests: round-trip (set → get → equal), startup hydration,
  write failure logs warning but doesn't roll back the swap.

### Phase 4 — Frontend model picker

- `lib/stores/tagging/api.ts`: `fetchActiveTagger()`,
  `listTaggerModels()`, `swapTaggerModel(selection)`.
- `lib/stores/tagging/taggerStatus.ts`: derive a `currentModel`
  from the `TaggerStatusEvent` source label (parse
  `hf:<repo_id>` / `local:<path>`).
- New section in `TagSettingsPanel.svelte`: model dropdown sourced
  from `listTaggerModels()`; "Currently running: …" indicator;
  "Swap" button disabled while any batch job is running, with a
  tooltip pointing at the stop button.
- "Loading…" state during the swap request (the request can take up
  to `tagger_response_timeout_seconds` on a slow in-flight call).
- Tests: stores (mocked fetch), panel snapshot (no real fetch).

## State machine

```
idle ──(request arrives)──> starting ──> ready
ready ──(swap with new identity)──> swapping ──> stopping ──> starting ──> ready
ready ──(swap with same identity)──> ready (no-op)
ready ──(swap while batch running)──> 409 (no state change)
swapping ──(respawn fails)──> failed ──(next request)──> starting ──> ready
```

`TaggerStatusEvent` carries the state + the new `source`. The
`swapping` state is transient but visible to other clients so they
know the active model is in flux.

## Risks / open questions

- **HTTP request duration during drain** — a long single-image
  inference (close to the 120s response timeout) will hold up the
  swap HTTP request for that long. Documented in the endpoint
  comment; UI shows a spinner. If this becomes painful, the swap
  endpoint could go 202 + SSE, but that adds complexity for an
  infrequent operation.
- **Concurrent swap requests** — `_lifecycle_lock` serializes them.
  A second swap arriving while the first is mid-drain waits in the
  lock queue; if the first one fails the second still proceeds with
  its own selection.
- **Custom HF repo_id** — SmilingWolf-only is fine for v1, but the
  UI already has a free-text input for repo_id in some places; a
  "Custom…" entry that surfaces the textbox would be a small
  follow-up.
- **Source label parsing** — the `taggerStatus` store derives
  `currentModel` from the event's `source` string. For v1, parsing
  is best-effort: empty source → "no model", `hf:<id>` → repo id,
  `local:<path>` → filename only. A more structured
  `TaggerStatusEvent.model` field would be cleaner but requires
  touching the event schema.