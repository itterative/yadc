---
date: 2026-06-20
---
# Refine mode design
**Context:** Phases 1–3 are shipped and the user has been testing
the MVP end-to-end. They reported it works "decent" and asked to
add a way to **refine existing templates** — start from one they
already have, ask the LLM to apply changes — rather than always
generating from scratch. This is a small extension of the existing
prompt generator (a second mode, not a new feature), and a bit
outside the original plan scope.

**Decision:** Add **Phase 6 (Refine mode)** to this plan.

- **UI** — `PillTabs` at the top of the form column with two
  tabs: "Generate" (current flow) and "Refine" (new flow). The
  streaming preview stays on the right, always visible. Reuses
  the existing `PillTabs` component (`lib/components/ui/tabs/`).
- **Refine form** — adds a "Template" section between Environment
  and Intent: a `<select>` picker over `$templates.items` plus
  a `JinjaEditor` (with variables chip strip, matching the
  preview's styling). The "Intent" textarea is relabelled to
  "Refinement intent" in refine mode and reuses the same
  persisted `promptSettings.intent` field. The bottom action
  button relabels to "Refine" in refine mode.
- **Backend** — `PromptGenerationRequest` gains an optional
  `template_content: str | None` field. When set, the service
  switches to refine mode. **No separate `mode` flag** — the
  presence of the field is the signal (keeps the API surface
  small and avoids a generate/refine flag mismatch). Pydantic
  validates that the field, if provided, is a non-empty string.
- **LLM message structure** for refine mode:
  - **0 examples** — fold the existing template + intent into a
    single user message (the 0-example case has no
    "before-the-examples" slot to insert into).
  - **N ≥ 1 examples** — reframe the first user message to
    mention refining an existing template, then insert a new
    user message with the existing template content + a new
    fake assistant ack (`"Got it. Send the next item."`)
    positioned right after the priming and before the first
    example. Existing example flow is unchanged; only the final
    "go" message gains "refined" wording. The generate-mode
    message structure is preserved verbatim, so Phase 2/3 tests
    keep passing.
- **System prompt** — new `_REFINE_SYSTEM_PROMPT` constant
  emphasising: the user has provided an existing template
  (apply changes, don't invent), preserve parts that work,
  output the full refined template (not a diff), keep the same
  Jinja2 structure rules as the generate system prompt.
- **Persistence** — only `mode` (generate/refine) is persisted
  in `promptSettings` ($version 2 with a `migrate` helper so
  v1 users keep their env / apiUrl / intent / focus). The
  template picker selection and the user's edits to the source
  template are **ephemeral** (re-select on reload to reload
  from the API). Same trade-off as the deferred IndexedDB
  follow-up for examples.
- **Save flow** — unchanged. Always uses the current
  "Save as new" flow. "Save over original" is a separate
  cross-cutting change to `EditTemplateDialog` that would
  apply to both generate and refine modes; deferred.

**Rationale:**
- Pill tabs match the user's preference and reuse the existing
  `PillTabs` component (no new UI primitives).
- The new-message-before-examples pattern keeps the existing
  priming flow intact. For 0 examples, folding into the first
  user message is the cleanest fit (the 0-example case has no
  slot for a separate message, and the priming pattern doesn't
  apply). Same controlled-component pattern for `mode` as the
  Phase 3b work — local `$state` initialised from the store,
  single `$effect` to sync back, no `bind:prop={$store.field}`.
- Persisting only `mode` keeps the persistence schema small;
  the user's edits are large and ephemeral (re-editing is
  cheap). The v1→v2 migration uses `storable`'s `migrate`
  argument (already supported, currently unused by any
  store) so v1 users don't lose their existing settings on
  upgrade.
- "Save as new" is consistent with the current flow and
  avoids accidental overwrites. Manual overwrite via
  `/templates` is available if the user wants to replace
  the original.

**Deferred follow-ups:**
- "Save over original" for the refined output (cross-cutting
  EditTemplateDialog change, would apply to generate mode
  too).
- Persisting the user's edits to the source template
  (similar to the deferred IndexedDB follow-up for examples).
- Persisting `lastTemplateName` so the user comes back to
  the same template selection.
- UI tests for the new form (matches the existing pattern
  — Phase 3 components have no vitest coverage).
- CLI parity for refine mode (`--refine <name>` +
  `--template-content <file>` flags, mirroring the
  `dataset:<name>:<n>` examples flag from Phase 4). Defer
  until Phase 4 lands for the base generate flow.

**Files touched:** listed in the plan's "Phase 6 (refine mode,
planned)" section under "Key Files Touched". No implementation
yet — this entry is the design record.
