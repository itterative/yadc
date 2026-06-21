---
date: 2026-06-20
---
# Phase 6b — Refine-mode frontend
**Context:** Phase 6a shipped the backend support for refining an
existing template (`template_content` field on `PromptGenerationRequest`,
`request.template_content is not None` switches mode). Phase 6b is the
matching frontend: a Generate/Refine pill switcher in the form column,
a template picker + `JinjaEditor` in refine mode, and persistence of
the `mode` choice (so the user comes back to the same tab).

**Decision:** Implemented per the Phase 6 design (history entry 004)
with three deviations, each chosen explicitly with the user:

1. **Form is rendered in BOTH `PillTabs` tab children** (not a custom
   pill bar). The plan said "PillTabs host"; the user picked this
   over the custom-bar option when I flagged the layout issue (PillTabs'
   hardcoded `flex-1` on the tabpanel doesn't suit a header-only
   switcher). The trade-off: both form instances are mounted (the
   inactive one is `hidden` via CSS), so the form's effects
   (env-loading, model-fetching, template-listing) run twice. The
   debounced store helpers dedupe in practice, so the double-fetch is
   a no-op for the store-backed calls. The forms share state via
   `bind:value` against the host's `promptSettings` snapshot, so
   env / apiUrl / intent / focus / examples / templateContent are
   always in sync. Internal state per instance (loading flags,
   template picker selection) re-initializes on tab switch — the
   `templateContent` itself is preserved (via bind), but the user has
   to re-pick from the dropdown to re-trigger the content fetch.
2. **Fixed a latent `storable.js` bug** to make the `migrate`
   parameter work. Before: `migrate(storedDataObj, version)` was
   called, the result assigned to `storedDataObj`, then the function
   exited without ever calling `store.set(...)` — so the migrated
   data was never written to the in-memory store or localStorage.
   Effectively a no-op. Fix: call `store.set(storedDataObj)` after
   the migrate returns (the existing subscribe handler then
   persists the result to localStorage). Without this fix, the v1→v2
   migration in `settings.ts` would have been a silent no-op and v1
   users would have lost their env / apiUrl / intent / focus on
   upgrade. Also added the matching
   `@typescript-eslint/no-unused-vars` config with
   `argsIgnorePattern: '^_'` and `varsIgnorePattern: '^_'` so the
   required-but-unused `_version` parameter in `migrateToV2` doesn't
   need an inline eslint-disable.
3. **`templateContent` is owned by the host** (`PromptGenerator.svelte`),
   not the form. The form receives it as `$bindable` so it can
   read/write the editor content (and the variables extraction stays
   form-local), but the host snapshots it for `startGeneration` and
   controls the "ephemeral, not persisted" policy in one place.

**Files touched:**
- `yadc/webui/src/lib/storable.js` — fixed the migrate bug
- `yadc/webui/eslint.config.js` — added `argsIgnorePattern: '^_'` /
  `varsIgnorePattern: '^_'` to `@typescript-eslint/no-unused-vars`
- `yadc/webui/src/lib/stores/prompts/settings.ts` — `mode` field,
  `$version: 2`, `migrateToV2` helper, `PromptGenMode` type
- `yadc/webui/src/lib/stores/prompts/types.ts` — `template_content`
  on `PromptGenRequest`
- `yadc/webui/src/lib/stores/prompts/actions.ts` — pass through
  `templateContent` to the request
- `yadc/webui/src/lib/components/prompts/PromptGenerator.svelte` —
  `PillTabs` host with form in both tabs, `mode` controlled-component
  wiring, `templateContent` local state, button relabel
  (Generate/Refine), `canGenerate` adds refine-mode validation
- `yadc/webui/src/lib/components/prompts/PromptForm.svelte` —
  `mode` and `templateContent` props, Template section
  (picker + `JinjaEditor` + variables chip strip) in refine mode,
  intent label/help text/placeholder switch in refine mode

**Checks:** 50/50 vitest tests pass, 724/724 backend pytest tests
pass, ESLint + Prettier + svelte-check + `npm run build` all clean.

**Deferred follow-ups (unchanged from the plan):**
- "Save over original" for the refined output (cross-cutting
  `EditTemplateDialog` change, would apply to generate mode too)
- Persisting the user's edits to the source template (similar to
  the deferred IndexedDB follow-up for examples)
- Persisting the template picker selection across tab switches
  (per-instance internal state — the user re-picks to re-fetch)
- UI tests for the new form (matches the existing pattern — Phase 3
  components have no vitest coverage)
- CLI parity for refine mode (`--refine <name>` +
  `--template-content <file>` flags; defer until Phase 4 lands
  for the base generate flow)
- Meta-prompt tuning based on real generation results
