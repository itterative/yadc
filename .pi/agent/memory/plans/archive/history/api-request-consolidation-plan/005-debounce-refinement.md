---
date: 2026-05-28
---
# Debounce Refinement: Additional APIs + Configurable Default

**Context:** After applying `debounce` to config and dataset images in Phase 2, two more APIs were identified as having duplicate calls in logs: `POST /api/envs/{name}/models` (fired on every env selection change in `EnvSelector`) and `GET /api/templates` + `GET /api/templates/{name}` (fired independently by `DatasetConfig`, `CaptionSettings`, and template management components).

**Changes:**
1. Added `debounce` wrappers to `fetchModels`, `fetchTemplates`, and `fetchTemplate` in their respective store modules
2. Changed the `debounce` default from 10 ms → **25 ms** via a single env variable: `PUBLIC_API_DEFAULT_DEBOUNCE_MS` in `yadc/webui/.env`
3. Added `ImportMetaEnv` / `ImportMeta` type declarations in `app.d.ts` so `import.meta.env.PUBLIC_API_DEFAULT_DEBOUNCE_MS` is typed
4. Removed all explicit `delay` arguments from debounced functions so they use the shared default

**Rationale:** 25 ms is short enough to feel instant to users but long enough to catch the duplicate calls fired by `$effect` hooks in sibling components mounted simultaneously. Making it an env var means it can be tuned without touching code.

**Files touched:**
- `yadc/webui/.env`
- `yadc/webui/src/app.d.ts`
- `yadc/webui/src/lib/async.ts`
- `yadc/webui/src/lib/stores/configs.ts`
- `yadc/webui/src/lib/stores/datasetImages.ts`
- `yadc/webui/src/lib/stores/envs.ts`
- `yadc/webui/src/lib/stores/templates.ts`
