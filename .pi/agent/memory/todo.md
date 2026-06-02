---
name: todo
description: Deferred tasks and improvements not tied to the current change.
---

# TODO

## Convert API backend to async (ASGI)

SSE (Server-Sent Events) is fundamentally incompatible with synchronous WSGI servers. SSE connections block a thread indefinitely, causing:
- **Ctrl+C hangs**: waitress and gunicorn both struggle to shut down with active SSE generators — the worker thread is stuck, and `SIGTERM`/`SIGINT` can't cleanly interrupt it. Gunicorn falls back to SIGKILL after `graceful_timeout`.
- **TCP proxy zombie connections**: If a TCP proxy buffers data after the real client disconnects, the backend has no signal the connection is dead. No WSGI server can detect this.
- **Thread pool exhaustion**: Each SSE connection consumes a thread for its entire lifetime.

The proper solution is to convert the API to an async framework (FastAPI/Starlette + uvicorn) where SSE connections are `async for` generators that can be cancelled by `asyncio` shutdown, and idle connections don't block threads.

This is a large refactor — Flask → FastAPI, injector DI integration, CORS middleware, all controllers, the SSE event system, etc. See stash `Gunicorn + SSE shutdown improvements` and `SSE subscriber TTL and yield stall detection` for intermediate workarounds that were attempted.

Key files that would change: `yadc/api/application.py`, `yadc/api/controllers/`, `yadc/api/modules/sse_events.py`, `yadc/api/events.py`, `pyproject.toml` (swap waitress → uvicorn, flask → fastapi).

## Frontend remaining cleanup

- SidePanel should accept a `class` prop — its positioning (inline vs fixed, width, etc.) is controlled by the parent page, not internal to the component
- ~~`isMobile` should be extracted into a reusable store~~ **DONE** — Replaced with CSS media queries (Tailwind `lg:` responsive prefixes). No JS resize listener needed.

## Frontend icons cleanup

**DONE** — All inline SVGs extracted into icon components:
- `SvgChevronLeft`, `SvgUpload`, `SvgSettings`, `SvgChat`, `SvgPhoto` (new)
- `SvgClose` (reused for close panel)
- All route files now use component imports, no inline SVGs remain
- All icons use Material Symbols style (fill-based, viewBox `0 -960 960 960`)
- Only `Checkbox.svelte` has an inline SVG (checkmark — acceptable UI primitive)

## Missing webui assets

- ~~**Favicon** — no favicon is set, browser shows a default blank tab icon~~ **DONE** — Favicons and web manifest added to `static/`, referenced from `app.html`, Flask serves them via `/<filename>` catch-all route in `app_frontend.py`.
- **Logo** — no app logo exists for use in the navbar/header and potentially as a larger brand mark. Theme-matching to be done later.

## Webui code quality pass

The webui frontend code is newly written and needs a cleanup pass to bring it up to a higher standard:

- **Error handling**: Consistent error boundaries, user-facing error messages (toast/banner instead of silent failures), proper handling of API error responses in stores
- **Models/types**: Review and tighten TypeScript types — ensure API response shapes are well-typed, avoid `any`, consider Zod schemas for API response validation (already used for SSE events)
- **Code style**: Consistent patterns for component structure, prop definitions, store usage, `$effect` cleanup, etc. Reduce duplication across similar components (e.g. the various dialog components)
- **General cleanup**: Remove dead code, unused imports, consolidate shared logic
- **Basedpyright warnings**: Fix all type-checking warnings in the webui-related backend code (and elsewhere)
- **Watchdog Observer type**: `Observer` from `watchdog` is currently typed as `Any` to suppress basedpyright errors — this needs a proper fix. Investigate why basedpyright can't resolve `watchdog.observers.Observer` (likely missing/incomplete stubs) and find the right solution (e.g. custom stub, `type: ignore` with comment, or wrap with a protocol)

## UI/UX overhaul

The current UI is functional but needs a manual pass to improve overall look and feel. This is a larger undertaking covering:

- Visual polish: spacing, typography, color consistency, border/shadow usage, hover/focus/active states
- UX improvements: better loading states, empty states, error feedback, confirmation prompts for destructive actions
- Responsive layout — ensure the UI works well at different viewport sizes
- Accessibility basics — keyboard navigation, ARIA attributes, focus management in dialogs
- Overall design coherence — the UI should feel like a unified application rather than assembled parts

## Simplified dataset config editing

The current EditDatasetDialog shows the raw TOML config, which is cumbersome for simple operations like adding/removing image paths. A simplified UI with individual path entries (add/remove buttons) would be much better. This requires:

- A `PATCH /configs/<name>` endpoint (or `PATCH /datasets/<name>`) that accepts granular operations like `add_path`, `remove_path`
- A frontend dialog with a clean list of paths, each with a remove button, and an add-new-path input at the bottom
- The TOML serialization should use multi-line strings for template values that contain newlines — currently they serialize as single-line strings with `\n` literals, which is unreadable
- Remove the old `ConfigEditor.svelte` from `SettingsDialog.svelte` — dataset config management is now handled through the dataset listing page (edit/delete buttons on cards, `EditDatasetDialog`)

## ~~Allow captioning images without a caption/draft~~ **DONE**

Images without captions/drafts are never filtered out — the filter only skips images where the target output already exists. The `overwrite` checkbox in the webui allows re-captioning already-captioned images.

## Test captioning flow in the webui

The full captioning workflow (start → progress → completion → result display) needs end-to-end testing through the webui to catch any integration issues between the frontend stores, SSE events, and the backend captioning API.

## Dataset browser scroll cutoff

The dataset browser grid uses `overflow-y-auto` on its parent div, but images near the bottom get cut off because the scroll container's padding doesn't extend past the last items. The fix is to replace the padding-based spacing on the scroll container with margin-based spacing on the children (grid items), so the last row of images is fully visible when scrolled to the bottom.

## ~~Centralize event handler registration~~ **DONE**

Services with `@event_handler` methods used to register themselves with `EventDispatcher` in their constructors. Now, `Application.configure_services()` iterates all services and calls `event_dispatcher.register_service(svc)` for each. Individual `register_service()` calls have been removed from service constructors.

## Watchdog job_id tag cleanup strategy

The `DatasetWatcherService._expected_sources` dict maps `dataset_name → job_id` so that `DatasetChangedEvent`s carry the `job_id` of the captioning operation that caused them. This lets the initiating frontend suppress the "Dataset files have changed" banner.

**Problem**: The tag is never cleared, so any future filesystem event for that dataset (e.g. adding a new image hours later) would carry the stale `job_id`. If that `job_id` is still in the frontend's 16-item list, the notification would be incorrectly suppressed.

**Challenge**: There's a fundamental race condition — we can't know when the *last* inotify event from a captioning session has been dispatched. Any timer-based clear is a heuristic:
- Too short: late inotify events arrive after the tag is cleared → untagged event → spurious notification on the initiating frontend
- Too long: legitimate external changes get tagged with a stale `job_id` → suppressed incorrectly

**Possible approaches**:
1. Timer-based clear with generous delay (e.g. 3-5x debounce interval). Pragmatic but racy.
2. Have `CaptioningService` listen for `DatasetChangedEvent` via `@event_handler` and re-tag them directly — avoids the watcher tag entirely but is architecturally complex.
3. Track a monotonic "captioning generation" per dataset in the watcher — `_dispatch_change` consumes the tag but stores it for a short grace period so the next event can still match.
4. Use a different mechanism entirely (e.g. per-request SSE filtering instead of event tagging).

The current implementation (never clearing) is being reverted. The job_id infrastructure in `DatasetChangedEvent`, `CaptioningService`, and the frontend `registerJobId`/bounded list should remain — only the watcher tag lifecycle needs a proper solution.

## SSE captioning event reliability / resumption

The frontend `CaptionProgress` component and dataset browser page derive captioning state from the global SSE `captioningStatus` store. If the final "done"/"error" event is missed (e.g. SSE reconnect, network blip, browser tab backgrounded), the UI gets stuck showing a running state forever.

Options:
1. **Periodic status events** — have the backend emit `CaptioningStatusEvent` at a regular interval (e.g. every 5-10s) while a job is running, so a reconnecting client quickly sees the current state
2. **SSE event resumption** — use `Last-Event-ID` (standard SSE mechanism) so the client can resume from where it left off after a reconnect. Requires the backend to assign IDs to events and buffer recent history
3. **Polling fallback** — add a `GET /datasets/<name>/caption/status` JSON endpoint (not SSE) that returns the current job snapshot. The frontend polls it as a fallback when no SSE events have been received for N seconds

Any of these also covers the case where the user opens the dataset page mid-captioning and the store hasn't received any events yet.

## Caption settings: dataset defaults integration

The caption settings panel currently uses hardcoded defaults and localStorage for persistence. The dataset's TOML config already has caption-relevant fields (`api.url`, `api.model_name`, `settings.max_tokens`, `settings.image_quality`, `reasoning.*`, `rounds`, `overwrite_captions`, `prompt.name/template`). These should be surfaced in the UI.

### Retrieve dataset config defaults

Add a backend endpoint (or extend the existing dataset info endpoint) that returns the caption-relevant fields from the dataset's TOML config — something like `GET /datasets/<name>/caption-defaults` returning a shape matching `CaptionJobOptions`. The frontend would load this alongside the dataset and use it to:
- Show what the dataset config specifies as defaults
- Pre-fill fields that have no localStorage override

### Two editing modes

**User settings mode (current default)**: Edit localStorage-backed settings. These override dataset config defaults. Show a visual indicator when a field differs from the dataset default (e.g. a small dot or italic "custom" badge next to the label, or a "Reset to dataset default" action on each field).

**Dataset config mode**: Edit the dataset's TOML config directly. This changes the defaults for all future sessions (and for CLI users). Should prompt the user to confirm since it modifies a shared config file. Could reuse the existing TOML editor or provide a structured form that writes back to the config.

UI approach: a toggle or mode switch at the top of the settings panel ("User Settings" / "Dataset Config"), or a contextual action (gear icon → "Edit dataset defaults…").

### Per-field reset

Each field that differs from the dataset default should have a small reset action (e.g. an X or ↩ icon) that clears the localStorage override and reverts to the dataset default. This requires the frontend to track both values (saved override vs. dataset default) simultaneously.

### Default source priority

When initializing the panel:
1. **Dataset config** provides the base defaults
2. **localStorage** overrides the dataset config for fields the user has explicitly changed
3. **Environment** provides API URL/token/model (already handled by EnvSelector)

Fields with no localStorage override and no dataset config value fall back to the hardcoded defaults (current behavior).

### Preset profiles

Let users save/restore named setting profiles (e.g. "Quick draft", "High quality") — basically named snapshots of the localStorage settings. Could be a dropdown at the top of the panel.

### Config diff indicator

Show a summary banner like "3 settings differ from dataset defaults" with a one-click reset-all.

## Incremental filesystem index updates

Currently, `DatasetChangedEvent` triggers a full `rescan_dataset()` which re-scans every file in every directory of the dataset. This is wasteful when only one file was added or removed.

**Fix**:
- Make `DatasetWatcherService` pass the affected file path(s) in the event (or a new granular event type like `DatasetFileAddedEvent` / `DatasetFileRemovedEvent`).
- `DatasetService` should handle these by doing targeted SQLite upserts/deletes for the affected files instead of a full rescan.
- Increase or remove the `_refresh_stale_datasets` interval (currently 60s) since the watcher now handles updates incrementally — the stale refresh becomes just a fallback.
