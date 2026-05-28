---
date: 2026-05-28
---
# Templates Page: Manual Refresh Pattern Analysis

**Context:** After adding SSE `TemplatesChangedEvent` for live sync, I reviewed `yadc/webui/src/routes/templates/+page.svelte` to understand the refresh pattern.

**Findings:**

The templates page uses **three distinct refresh paths**:

1. **`onMount` → `loadTemplates()`** — initial page load fetches the template list
2. **Post-CRUD manual refresh** — after `deleteTemplate()`, `handleTemplateSaved()`, and `handleTemplateCreated()`, the code calls `refreshTemplates()` directly
3. **SSE `templates_changed` → `refreshTemplates()`** — auto-triggered by filesystem watcher

**Why manual refreshes are kept alongside SSE:**

The backend `TemplateWatcherService` has a **1-second debounce**. If a user saves a template in `EditTemplateDialog`, the sequence would be:

| Time | Without manual refresh | With manual refresh (current) |
|------|----------------------|------------------------------|
| t=0 | User clicks Save | User clicks Save |
| t=50ms | API returns 200 OK | API returns 200 OK |
| t=100ms | UI still shows old state | `handleTemplateSaved()` → `loadTemplates()` → UI updated |
| t=1000ms | SSE fires, UI finally updates | SSE fires (no-op, already up to date) |

**Conclusion:** Manual refreshes after CRUD operations are required for **instant same-tab feedback**. SSE handles **cross-tab sync** and **external changes** (CLI/file edits). The debounce on `fetchTemplates` (25ms) ensures no duplicate HTTP calls when both paths overlap.

**Same pattern applies to:**
- Environments (`EnvSelector.svelte`, `EnvironmentSettings.svelte`)
- Datasets (`DatasetConfig.svelte` save operations)

**Decision:** Keep both mechanisms. Do not remove manual refreshes.
