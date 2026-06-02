---
name: todo
description: Deferred tasks and improvements not tied to the current change.
---

# TODO

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

The webui is missing basic assets:
- **Favicon** — no favicon is set, browser shows a default blank tab icon
- **Logo** — no app logo exists for use in the navbar/header and potentially as a larger brand mark

These should be added to `static/` (SvelteKit's static asset dir) and referenced from the app layout.

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

## Centralize event handler registration

Services with `@event_handler` methods currently register themselves with `EventDispatcher` in their own constructors (e.g. `SSEEvents`, `DatasetService` call `event_dispatcher.register_service(self)`). This is fragile — easy to forget when adding handlers to a new service.

**Fix**: In `Application.configure_services()`, after instantiating all services, iterate through `self.services` and call `event_dispatcher.register_service(svc)` for each. Then remove the manual `register_service()` calls from individual service constructors.

## Incremental filesystem index updates

Currently, `DatasetChangedEvent` triggers a full `rescan_dataset()` which re-scans every file in every directory of the dataset. This is wasteful when only one file was added or removed.

**Fix**:
- Make `DatasetWatcherService` pass the affected file path(s) in the event (or a new granular event type like `DatasetFileAddedEvent` / `DatasetFileRemovedEvent`).
- `DatasetService` should handle these by doing targeted SQLite upserts/deletes for the affected files instead of a full rescan.
- Increase or remove the `_refresh_stale_datasets` interval (currently 60s) since the watcher now handles updates incrementally — the stale refresh becomes just a fallback.
