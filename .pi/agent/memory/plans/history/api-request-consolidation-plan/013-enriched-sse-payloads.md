---
date: 2026-05-28
---

## Enriched SSE Event Payloads

Changed `EnvironmentsChangedEvent` and `TemplatesChangedEvent` from empty `{}`
to include the list of changed names. Also refactored the watcher base class to
accumulate changed file paths during debounce and delegate event creation to
subclasses.

### Changes

1. **`SinglePathWatcherService` (`watcher_base.py`)** — Removed `event_factory` parameter.
   Added `create_event(changed_files: frozenset[str]) -> Event` method for subclasses
   to override. The debounce now accumulates file paths in `_pending_files` and passes
   them to `create_event()` when the timer fires.

2. **`_FilterEventHandler`** — Now passes the changed file path (as `str`) to the
   `on_change` callback instead of calling with no arguments.

3. **`EnvironmentsChangedEvent`** — Added `envs: list[str]` field. Since all envs
   live in a single `config.toml`, the event lists all current env names (from
   `cmd_envs.list_all_env()`).

4. **`TemplatesChangedEvent`** — Added `templates: list[str]` field. Lists all current
   template names (via `cmd_templates.list_user_template()`) to match the envs behavior —
   consistent full-list payload so the frontend can unambiguously detect additions and deletions.

5. **Frontend Zod schemas** — `EnvironmentsChangedEventZ` now validates `{ envs: string[] }`,
   `TemplatesChangedEventZ` now validates `{ templates: string[] }`.
