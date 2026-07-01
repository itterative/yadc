---
date: 2026-07-01
---
# Async (202 + SSE) swap + split startup timeout

**Context:** A slow first-run tagger model download (HuggingFace fetch
inside the worker's `load_model`) was getting cancelled. Two deadlines
conspired: (1) `TaggerServer.start()` shared one `response_timeout`
(120s) for *both* the model-load phase and per-tag inference, so a slow
download blew the startup wait and `start()` killed the worker
mid-download; (2) even with a larger startup cap, the `/tagger/swap`
HTTP handler awaited the whole drain+download+respawn, so Quart's
60s `response.timeout` cancelled the coroutine before the download
finished. The `hf_xet` package was already added for resumable
downloads but didn't fix the cancellation itself.

**Decision:**
1. **Split the timeout.** `TaggerServer` (and `TaggerClient`) gained a
   separate `start_timeout` (default `STARTUP_TIMEOUT_SECONDS = 900s`,
   configurable via `Configuration.tagger_startup_timeout_seconds`).
   `start()` is bounded by `start_timeout`; `tag()` keeps
   `response_timeout`. The startup phase (download) is now allowed to
   run far longer than a single inference.
2. **Make the swap fire-and-forget (202 + SSE).** `swap_active_model`
   now does synchronous validation (no-op / busy → 409 / swap-in-progress
   → 429) and returns the intended selection immediately; the
   drain→stop→flip→respawn→persist runs in a background `asyncio.Task`
   (`_swap_background`, held alive by `self._swap_task`). The controller
   returns **202 Accepted**. The existing `tagger_status` SSE stream
   already emits the full transition (stopping/stopped with the *old*
   label, then starting/ready/failed with the *new* label), so the
   frontend refreshes on state change without a long-hanging POST.
   Failures in the detached task are reported via the `failed` SSE
   event (not a raised exception) and roll back `_active_tagger`.

**Rationale:** This is exactly the "202 + SSE" option the original plan
deferred as too complex for an infrequent operation. The download-
cancellation bug made it worth the complexity: the HTTP handler no
longer blocks on network I/O it can't control, and the worker's larger
startup deadline isn't fighting the request handler's deadline.

**Frontend:** `swapActiveModelAction` no longer toasts success on 202
(the swap is still running); `GeneralSettings.svelte` subscribes to the
`taggerStatus` store, keeps the spinner up while `swapInFlight`, and
toasts success/failure when the SSE state settles. A `sawSwapTransition`
guard ignores the stale pre-swap status so an already-ready subprocess
doesn't short-circuit the wait. On `failed` it re-syncs `active` from
`GET /api/tagger/active` (rollback happened server-side).

**Files touched:**
- `yadc/taggers/server.py` — `STARTUP_TIMEOUT_SECONDS`, `start_timeout`
- `yadc/taggers/client.py` — `start_timeout` passthrough
- `yadc/api/configuration.py` — `tagger_startup_timeout_seconds`
- `yadc/api/services/tagging.py` — `swap_active_model` (async) +
  `_swap_background` + `_swap_task` ref
- `yadc/api/controllers/api_tagging.py` — 202 + docstring
- `yadc/webui/src/lib/stores/tagging/actions.ts` — drop premature toast
- `yadc/webui/src/lib/components/settings/GeneralSettings.svelte` — SSE-driven swap completion
- Tests: `tests/taggers/test_server.py` (split-timeout), `conftest.py`
  (`run_swap` helper), `test_swap_active_model.py` / `test_service.py`
  (await background task), `test_tagger_swap_endpoints.py` (202)
