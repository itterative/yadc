---
date: 2026-06-03
---
# Phase 1: introduce 5 domain sub-folders in `lib/stores/`

**Context:** All 16 store files (~2,233 lines) sat in a flat `src/lib/stores/` directory with no grouping by domain. The 615-line `datasetImages.ts` was already deferred from the prior component-org plan for splitting. The new `plans/archive/frontend-store-organization.md` plan was approved and called for grouping stores by domain (dataset, caption, config, env, templates), with the 7 global UI singletons and `events.ts` staying at top level.

**Decision:** Per the plan, one step per domain:

- **`dataset/`** — split the 615-line `datasetImages.ts` into `types.ts` (~75 lines, 12 types) + `api.ts` (~540 lines, all fetch/CRUD/upload/captioning helpers) + `index.ts` (re-exporting barrel). 18 import sites updated from `$lib/stores/datasetImages` → `$lib/stores/dataset`.
- **`caption/`** — moved `captionActions.ts` (88), `captionOptions.ts` (21), `captionSettings.ts` (33) into `actions.ts` + `options.ts` + `settings.ts`. 7 import sites updated. Updated internal cross-folder imports: `./datasetImages` → `../dataset/api`, `./events` → `../events`, etc. Dropped the deprecated `captioning.ts` re-export shim (0 callers).
- **`config/`** — split the 257-line `configs.ts` into `types.ts` (~95 lines, all config + export + draft types) + `api.ts` (~120 lines, all config CRUD + export + drafts API). 7 import sites updated.
- **`env/`** — split the 153-line `envs.ts` into `store.ts` (~50 lines, types + `envs` writable + `refreshEnvs`) + `api.ts` (~110 lines, env CRUD + model fetching + key-mode). 4 import sites updated (2 in initial pass, 2 in env/ component subfolder caught by svelte-check).
- **`templates/`** — split the 119-line `templates.ts` into `store.ts` (~40 lines) + `api.ts` (~55 lines) + `jinja.ts` (~30 lines, the `extractVariables` pure helper). 0 import-site updates needed (path was already `$lib/stores/templates`).

**Deviations from the plan:**

- **Combined steps 1.1 + 1.2 into a single commit boundary.** Step 1.1 alone left `captionActions.ts` with broken imports (`./datasetImages` no longer exists). The plan's "one commit per step" assumption broke because of the cross-folder import. Solution: did all the file moves and import rewrites first, then verified with `svelte-check` only at the end of 1.2 (and at each subsequent step's end). Commits land with the codebase in a working state at each commit boundary, not at each plan step.
- **Missed 2 import sites in `lib/components/env/` on the first pass.** The env components import from `$lib/stores/envs`, but my initial `rg -l` returned 2 files (events.ts + SecuritySettings.svelte) — the env components were missed because the search pattern matched `from '\\$lib/stores/envs'` literally and the `env/` component path confused the match. Caught by `svelte-check` after the env `git rm`. Fix was a targeted `sed -i 's|stores/envs|stores/env|g'` on the two env components.
- **`captioning.ts` dropped** as planned (0 callers, confirmed via `rg`).
- **Re-export concerns in `env/api.ts`**: Initially added `export type { EnvInfo, EnvListResult } from './store';` at the bottom of `api.ts` to ensure type re-export through the barrel. Removed it after review: `index.ts` re-exports both `store` and `api`, so types are reachable via `$lib/stores/env` regardless. The `import type` at the top of `api.ts` is sufficient (and is a type-only import, so no runtime circular dependency).

**Result:**
- `src/lib/stores/` now has 5 sub-folders + 7 top-level files + `events.ts`. From 16 flat files to a structured layout.
- `lib/stores/dataset/api.ts` is ~540 lines — still large but every function is in scope. Further splitting is a possible follow-up.
- All import sites use the new domain paths. No shim files needed (the `index.ts` barrels do the re-exporting).

**Verification:** `npx svelte-check` → 0 errors / 0 warnings. `npx eslint --fix .` → clean. `npx prettier --write .` → no changes. `npm run build` → success. User confirmed visual smoke test outside tmux.

**Files touched:**
- Created: `dataset/{types,api,index}.ts`, `caption/{actions,options,settings,index}.ts`, `config/{types,api,index}.ts`, `env/{store,api,index}.ts`, `templates/{store,api,jinja,index}.ts`
- Deleted: `datasetImages.ts`, `captionActions.ts`, `captionOptions.ts`, `captionSettings.ts`, `captioning.ts`, `configs.ts`, `envs.ts`, `templates.ts`
- Updated: 36 import sites across `src/lib/components/`, `src/routes/`, and `src/lib/stores/events.ts`
