---
date: 2026-05-28
---
# Template SSE Sync

**Goal:** Apply the same thin-event SSE pattern from Phase 1.5 (env changes) to templates.

**Context:** Templates are stored as individual `.jinja` files in `STATE_PATH/templates`, not in `config.toml`. This required a separate filesystem watcher from the env watcher.

**Backend changes:**
- Added `TemplatesChangedEvent` to `yadc/api/events.py`
- Added `@event_handler(TemplatesChangedEvent)` to `SSEEvents` in `sse_events.py`
- Created `TemplateWatcherService` in `yadc/api/modules/template_watcher.py` using `watchdog` to monitor `STATE_PATH/templates` for `*.jinja` changes
  - 1-second debounce
  - Gracefully skips if the templates directory doesn't exist yet (e.g., first run)
- Exported `TemplateWatcherService` from `modules/__init__.py` for auto-discovery

**Frontend changes:**
- Added `TemplatesChangedEventZ` schema to `stores/events.ts`
- Subscribed to `templates_changed` events in the SSE `connect()` listener and called `refreshTemplates()` automatically

**Benefits:**
- Multi-tab sync: editing templates in one tab updates all other tabs automatically
- CLI changes detected: editing `.jinja` files directly or using a future CLI command will push live updates
- Frontend `fetchTemplates` is debounced (25ms), so overlapping manual refresh + SSE refresh collapse to one request

**Files touched:**
- `yadc/api/events.py`
- `yadc/api/modules/sse_events.py`
- `yadc/api/modules/template_watcher.py` (new)
- `yadc/api/modules/__init__.py`
- `yadc/webui/src/lib/stores/events.ts`
