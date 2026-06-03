---
date: 2026-05-28
---
# Phase 1.5: SSE Environment Change Events

**Goal:** Push environment list updates to all connected clients via SSE so frontends stay in sync without manual refreshes.

**Key decision:** Use a filesystem watcher on `config.toml` (where environments are stored) instead of dispatching from API controllers. This catches both webui edits and CLI changes uniformly. The `EnvWatcherService` debounces changes at 1 second to coalesce rapid bursts.

**Backend changes:**
- Added `EnvironmentsChangedEvent` to `yadc/api/events.py`
- Added `@event_handler(EnvironmentsChangedEvent)` to `SSEEvents` in `sse_events.py`
- Created `EnvWatcherService` in `yadc/api/modules/env_watcher.py` using `watchdog` to monitor `CONFIG_PATH/config.toml`
- Exported `EnvWatcherService` from `modules/__init__.py` for auto-discovery by the DI framework

**Frontend changes:**
- Added `EnvironmentsChangedEventZ` schema to `stores/events.ts`
- Subscribed to `environments_changed` events in the SSE `connect()` listener and called `refreshEnvs()` automatically
- Removed the manual "reload when settings dialog closes" effect from `EnvSelector.svelte` — now redundant thanks to live sync

**Benefits:**
- Multi-tab sync: editing envs in one tab updates all other tabs automatically
- CLI changes detected: running `yadc envs set ...` from the terminal pushes a live update to all open webui clients
- Frontend `fetchEnvs` is debounced (25ms), so overlapping manual refresh + SSE refresh collapse to one HTTP request

**Files touched:**
- `yadc/api/events.py`
- `yadc/api/modules/sse_events.py`
- `yadc/api/modules/env_watcher.py` (new)
- `yadc/api/modules/__init__.py`
- `yadc/webui/src/lib/stores/events.ts`
- `yadc/webui/src/lib/components/settings/EnvSelector.svelte`
