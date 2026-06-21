---
date: 2026-06-20
---
# Phase 6b revision — combined Generate/Refine form + inline mode toggle

**Context:** Phase 6b (see `008-phase-6b-implementation.md`) shipped
refine mode as a Generate/Refine `PillTabs` switcher, rendering
`PromptForm` in **both** tab children (the inactive one CSS-hidden).
This was an accepted trade-off: both form instances mount, so their
effects fire twice and per-instance internal state resets on tab
switch. In practice the main pain was that the **env selector was
mounted twice** (along with the model-list and template-list fetches),
and the template-picker selection was lost every time the user
switched between Generate and Refine.

**Decision:** Collapse the Generate/Refine tab switcher into a single
form with an inline mode toggle. `PillTabs` goes from three tabs
(Generate | Refine | History) to two (Compose | History), and the
form is rendered **once** in the Compose tab.

- **Mode toggle**: a `New` / `Refine` pill group (mirroring the
  existing Focus pill group) placed **between Focus and the Few-shot
  examples**. Selecting `Refine` reveals the Template section (picker
  + `JinjaEditor` + variables chip strip) right there. The Template
  section moved out from between Environment and Intent to sit under
  the mode pills.
- **`mode` unchanged as a value**: still `'generate' | 'refine'`
  (maps `generate` → "New", `refine` → "Refine"). No change to
  persistence (`promptSettings` stays `$version: 2`), the History
  types, or the backend `template_content` field — the toggle just
  reads/sets `mode`.
- **`PromptForm.mode` is now `$bindable`**; the host binds `bind:mode`.
  The inline pills update it, which propagates to the host's `mode`
  `$state` and thus the footer button label (`Generate`/`Refine`),
  `canGenerate` / `canSave`, the `startGeneration` /
  `saveHistoryEntry` snapshots, and the persistence `$effect`.
- **`activeTab` simplified** from `'generate' | 'refine' | 'history'`
  to `'compose' | 'history'`. The `activeTab → mode` sync `$effect`
  and the `untrack(() => mode)` initialiser are removed — `mode` is
  now driven solely by the form pills and by History restore. `mode`
  no longer needs to be derived from the active tab.
- **`handleRestore`** switches to the Compose tab (`activeTab =
  'compose'`) and sets `mode = entry.mode` plus the fields.

**Why not a custom pill bar (the alternative floated in 008):** the
mode toggle now lives *inside* the single form, so `PillTabs` is only
used for its real job — switching between the form and the History
panel. Keeping `PillTabs` for that retains the accessibility wiring
(role="tablist", aria-selected) without the double-mount cost.

**Files touched:**
- `yadc/webui/src/lib/components/prompts/PromptGenerator.svelte` —
  one `<PromptForm>` instance, `bind:mode`, `activeTab` type +
  init simplified, mode-sync `$effect` removed, `handleRestore` uses
  `'compose'`, comment blocks updated.
- `yadc/webui/src/lib/components/prompts/PromptForm.svelte` — `mode`
  is `$bindable`, added the New/Refine pill section, moved the
  Template section under it, cleaned up the now-stale "re-mounts on
  tab switch" comment on the env-change `$effect`.
- `.pi/agent/memory/docs/frontend-patterns.md` — replaced the
  "Content placement (form-in-both-tabs pattern)" subsection (the
  pattern is gone) with a short "don't duplicate a component across
  tabs; use an inline control" note.
- `.pi/agent/memory/docs/frontend/components-domain.md` — updated the
  `PromptGenerator.svelte` and `PromptForm.svelte` summaries.
- `.pi/agent/memory/docs/frontend/stores.md` — updated the `mode`
  persistence wording (New/Refine selection, not a tab).

**Checks:** ESLint + Prettier + svelte-check + `npm run build` all
clean; 13/13 prompts vitest pass. Backend untouched (no Python
changes).

**Resolves:** the "template picker selection lost on tab switch"
deferred follow-up from `008-phase-6b-implementation.md` — with a
single form instance the picker selection survives a mode switch.
The "form-in-both-tabs pattern" doc subsection is removed (no longer
used anywhere in the codebase).
