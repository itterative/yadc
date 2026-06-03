---
date: 2026-05-28
---
# Phase 1: Environment Details Endpoint

**Context:** `GET /api/envs` used to return `list[str]`, forcing `EnvSelector.svelte` to call `GET /api/envs/{name}` on every selection change. The settings dialog (`EnvironmentSettings.svelte`) also loaded env names first, then fired N parallel `fetchEnv()` calls to build an `envDetailMap`.

**Decision:** Changed `GET /api/envs` to return `list[EnvInfo]` directly (Option B). Extracted `_format_env()` helper in the backend to share formatting logic with `GET /envs/<name>`. Frontend `envs` store now holds full objects. `EnvSelector` derives `envInfo` from the store reactively. `EnvironmentSettings` removed `envDetailMap` entirely and receives `EnvInfo` directly in `startEditEnv()`.

**Rationale:** Env payloads are tiny (<10 envs typical), so the "wasteful payload" concern is theoretical. The simpler API surface and elimination of N+1 requests is a real win. No consumers outside the webui were relying on the old `list[str]` shape.

**Files touched:**
- `yadc/api/controllers/api_envs.py`
- `yadc/webui/src/lib/stores/envs.ts`
- `yadc/webui/src/lib/components/settings/EnvSelector.svelte`
- `yadc/webui/src/lib/components/dialogs/EnvironmentSettings.svelte`
