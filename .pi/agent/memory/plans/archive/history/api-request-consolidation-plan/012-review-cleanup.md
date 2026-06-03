---
date: 2026-05-28
---

## Cleanup: Review Findings

Code review of the 22 temporary commits (22ef40f..59b49f1) identified several issues.

### Changes Made

1. **`debounce()` rewritten** (`yadc/webui/src/lib/async.ts`) — Fixed to properly reset the timer on each call (true debounce) instead of only deduping. Added SSR guard for `!browser` pass-through.

2. **Watcher base class** (`yadc/api/watcher_base.py`) — Extracted `SinglePathWatcherService` from `env_watcher.py` and `template_watcher.py`. Both reduced from ~115 lines to ~38 lines. Shared logic: debounce, observer lifecycle, directory-existence check.

3. **`resumption_failed` handler** (`yadc/webui/src/lib/stores/events.ts`) — Added `refreshTemplates()` alongside existing `refreshEnvs()` call.

4. **True LRU in `_storedCaptions`** (`yadc/webui/src/lib/stores/events.ts`) — `getStoredCaption()` now promotes accessed entries. `MAX_STORED_CAPTIONS` reduced to 64 with a note that the feature may be refined.

5. **`.env.example`** (`yadc/webui/.env.example`) — Documents `PUBLIC_API_DEFAULT_DEBOUNCE_MS=25`.

### Deferred

- `timeout_keep_alive=10` in `application.py` — unrelated, owner will move to separate commit.
- `fetchEnv()` export — owner will investigate removal.
