---
date: 2026-06-20
---
# Phase 7 — Prompt-history persistence (implementation)
**Context:** Phase 7 design (`009-phase-7-design.md`) added a
server-side store for the prompt-generator artifact (intent +
focus + examples + mode + template content) so the user can
save and restore the full prompt across sessions. Implementation
went per the design with three minor deviations, all chosen
explicitly with the user during the work.

**Decision: ``activeTab`` is a separate ``$state`` from ``mode``**.
The PillTabs now have three children (`Generate | Refine |
History`). ``mode`` is the form's mode — a `PromptGenMode`,
persisted to `promptSettings`, used by the form / button label
/ `canGenerate` check. ``activeTab`` is the tab the user is
currently viewing — UI-only, NOT persisted. When the user
clicks Generate or Refine, the `$effect` syncs `mode` to
match. When the user clicks History, `mode` is untouched
(so a restore in History mode can set `mode` to the entry's
mode without losing the user's last form mode). Initial
value of `activeTab` is the persisted `mode` so a returning
user lands on Generate/Refine, never History.

The init reads `mode` once via `untrack(() => mode)` — a
Svelte 5 idiom that suppresses the "reference only captures
the initial value" warning without subscribing to changes.
Plain `let activeTab = $state<ActiveTab>(mode)` would warn
(we don't want a reactive re-init every time `mode` flips).

**Decision: persist the full template content, drop the
`had_template` boolean (per user spec).** The list view
derives `had_template` from `template_content IS NOT NULL` —
included separately in the response so the frontend doesn't
have to know about the nullable field's semantics. Restore
loads the full template content for refine-mode entries
(generate-mode entries have `template_content = NULL` and
leave the editor empty). This makes "save in-progress
refine" work as a real cross-session use case — the user
explicity preferred this over a `had_template: bool` flag.

**Decision: form footer layout is `[examples count] [Save]
[Generate/Refine]` (in form tabs) or `[examples count]`
(when on History tab).** The History tab hides the action
buttons — the user is browsing, not generating. The
`Save to history` button is disabled until the same minimum
required for generate (intent + non-empty template in refine
mode). The button is in the form footer (next to Generate)
rather than at the top of the form, so it's grouped with the
primary action — the user said "Save is just a side button".

**JSON contract:**

```jsonc
// POST /api/prompts/history  request
{
  "mode": "generate" | "refine",
  "intent": "...",
  "focus": "system" | "user" | "both",
  "examples": [{"subject": "...", "caption": "...", "image_data_url": "data:..."}],
  "template_content": "..." | null
}

// GET /api/prompts/history  response (summary row)
{
  "id": 123,
  "mode": "generate" | "refine",
  "had_template": true | false,   // server-derived from template_content IS NOT NULL
  "focus": "...",
  "intent_preview": "...",        // first 200 chars (truncated with "…")
  "example_count": 2,             // server-derived (SQL aggregate over example rows)
  "created_t": 1700000000.0
}

// GET /api/prompts/history/<id>  response (full entry, for restore)
{
  "id": 123,
  "mode": "...",
  "intent": "...",
  "focus": "...",
  "examples": [...],
  "template_content": "..." | null,
  "created_t": 1700000000.0
}
```

**Files touched:**
- `yadc/api/migrations/0008_prompt_history_{up,down}.sql` (new)
- `yadc/api/services/prompt_history_repository.py` (new — `PromptHistoryEntry` + all SQL)
- `yadc/api/services/prompt_history.py` (new — service, `PromptHistorySaveRequest` Pydantic model, `PromptHistoryListItem` + `PromptHistoryPage` dataclasses, `INTENT_PREVIEW_MAX_CHARS = 200`, `_make_intent_preview` helper, `PROMPT_HISTORY_MAX_ENTRIES = 20`, `save_entry` (insert + prune in one transaction), `list_history` (limit+1 probe + JSON count), `get_entry` (Pydantic-validated decode), `delete_entry`)
- `yadc/api/controllers/api_prompts_history.py` (new — four endpoints; `SaveHistoryBody` Pydantic model)
- `yadc/api/services/__init__.py` (re-export `PromptHistoryEntry`, `PromptHistoryListItem`, `PromptHistoryPage`, `PromptHistoryRepository`, `PromptHistorySaveRequest`, `PromptHistoryService`, `PROMPT_HISTORY_MAX_ENTRIES`)
- `yadc/webui/src/lib/stores/prompts/types.ts` (add `PromptHistoryListItem`, `PromptHistoryEntry`, `SaveHistoryArgs`)
- `yadc/webui/src/lib/stores/prompts/api.ts` (add `fetchHistoryList`, `fetchHistoryEntry`, `saveHistoryEntry`, `deleteHistoryEntry`)
- `yadc/webui/src/lib/components/prompts/PromptHistoryPanel.svelte` (new — list view, restore button, delete with confirm dialog, refresh button, relative timestamps)
- `yadc/webui/src/lib/components/prompts/PromptGenerator.svelte` (add History tab + Save button + `handleSaveToHistory` + `handleRestore`; introduce `activeTab` + `untrack` init)
- `yadc/webui/src/lib/components/prompts/index.ts` (export `PromptHistoryPanel`)
- `tests/api/test_prompt_history_repository.py` (new — 17 tests, real DB)
- `tests/api/test_prompt_history_service.py` (new — 19 tests, mixed mocked + real)
- `tests/api/test_api_prompts_history.py` (new — 12 tests, controller-only with mocked service)

**Doc updates:**
- `plans/prompt-generator-plan.md` — Phase 7 section marked done, status updated
- `plan-management.md` — index status updated
- `docs/backend/api.md` — new service + controller entries
- `docs/frontend/components-domain.md` — new `PromptHistoryPanel` entry
- `docs/frontend/stores.md` — new history types + API functions
- `todo.md` — removed the superseded "IndexedDB for prompt-generator few-shot examples" entry

**Checks:** 772/772 backend tests pass (was 724, +48 new), 50/50 vitest tests pass (no new tests, mirroring the Phase 6b pattern of skipping vitest for feature components). ESLint, Prettier, svelte-check, ruff, ruff format, basedpyright, `npm run build` all clean.

**Test count details (Phase 7 only, +48):**
- Repository: 17 (round-trip, list ordering, limit/before_id, get/delete/count, prune cap edge cases)
- Service: 19 (save orchestration w/ mocked repo, list-view derivation incl. truncation + JSON count, get-entry round-trip incl. malformed JSON, delete)
- Controller: 12 (save body validation + 500-on-read-back-fail, list pagination, get 404, delete 404)

**Deviations from the design (and why):**
1. **`_validate_body` import decision (controller)**: Initially inlined a local `validate_body` to "avoid cross-controller import" — that was wrong, the canonical one in `utils_json` is fine. Cleaned up to use the canonical import. (Lesson: don't second-guess the existing utility pattern.)
2. **`mode` type vs `activeTab` separation**: Design sketch implied `mode` would absorb the History value. Splitting them is cleaner (no `history` in the persisted `PromptGenMode` enum) and matches the existing `PromptForm` props which expect `mode: 'generate' | 'refine'`. The user accepted the small added complexity (two states + sync effect) in exchange for keeping `PromptForm`'s prop type stable.
3. **Save button location**: Design said "in the form footer". Implemented there. Considered putting it in the page Topbar for prominence; rejected — Topbar is page-level, the form footer is action-level, and putting it next to Generate groups related actions.

**Deferred / out of scope (carried over from the design):**
- CLI parity for save/restore
- Per-entry "pin" / favorite
- Search/filter in the history list
- "Save over original" semantics for the refined output
