---
date: 2026-06-20
---
# Phase 7 — Prompt-history persistence
**Context:** The prompt generator currently keeps few-shot
`ExamplePair` payloads in memory only — reloading the page drops
them because the `image_data_url` blobs are too large for
localStorage. The deferred IndexedDB follow-up (in `todo.md`)
explored a local-storage solution but the user prefers server-side
persistence so the entries survive across machines / profile
switches and so the work can be shared with the CLI in a later
phase.

**Scope:** Add a "save current as a history entry" action to the
existing prompt generator, plus a "restore from history" action.
The user-stated scope is **the prompt itself** — intent, focus,
examples, mode, and the template content (if any). Env, model,
and the template picker selection are explicitly OUT of scope
(those are workspace-level concerns, not part of the artifact).

**Decision: scope of one saved entry**

| Field | Persisted? | Notes |
|-------|------------|-------|
| `mode` (`generate`/`refine`) | yes | Needed to know which tab to switch to on restore. |
| `intent` | yes | The artifact. |
| `focus` (`system`/`user`/`both`) | yes | The artifact. |
| `examples` (with images) | yes | The artifact. Stored as raw BLOBs in a sibling `prompt_example_images` table; the service base64-translates at the wire boundary. |
| `template_content` | yes | The full template body. NULL for generate mode, non-NULL for refine. The user explicitly rejected the simpler `had_template: bool` approach — saving the actual content makes "save in-progress refine" work as a real cross-session use case. |
| `template_name` (picker selection) | no | A user re-selects from the picker on restore. |
| `env`, `api_url`, `api_token`, `api_model_name` | no | User's workspace-level config, not part of the artifact. |

**Restore semantics (frontend-only, no special endpoint):**
- Set `mode` → switch to the matching tab.
- Set `intent`, `focus`, `examples` — local state of the form.
- If `template_content is not None`, set it as the editor content
  (refine mode only).
- Toast: "Restored from history".

**Schema (new table `prompt_history`):**

```sql
CREATE TABLE prompt_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL CHECK (mode IN ('generate', 'refine')),
    intent TEXT NOT NULL,
    focus TEXT NOT NULL CHECK (focus IN ('system', 'user', 'both')),
    template_content TEXT,               -- NULL for generate; non-NULL for refine
    created_t REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX idx_prompt_history_created ON prompt_history (created_t DESC);

CREATE TABLE prompt_example_images (
    entry_id INTEGER NOT NULL REFERENCES prompt_history(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    subject TEXT NOT NULL,
    caption TEXT NOT NULL,
    mime TEXT NOT NULL,
    data BLOB NOT NULL,                  -- raw bytes; service base64-translates at the wire boundary
    PRIMARY KEY (entry_id, ordinal)
);
CREATE INDEX idx_prompt_example_images_entry ON prompt_example_images (entry_id);
```

`had_template` is dropped — the list view derives it from
`template_content IS NOT NULL` (set on the backend in the response
so the frontend doesn't have to know about the field's semantics).

**Decision: save trigger = manual button.** Auto-save on generate
success was considered but rejected: it would accumulate abandoned
drafts the user doesn't care about, and gives the user no way to
discard a malformed intent before it lands in history. Manual
"Save to history" in the form footer is one click and keeps the
list clean.

**Decision: hard cap = 20, auto-prune oldest on save.** Matches
the `config_history` pattern (`DEFAULT_MAX_ENTRIES = 50` there).
Prune is a single SQL statement inside the same transaction as
the insert: `DELETE WHERE id NOT IN (SELECT id ORDER BY id DESC
LIMIT 20)`. The 20 newest are kept; everything else goes.

**Decision: UI = third tab in the existing form-card PillTabs**
(`Generate | Refine | History`). The History tab hides the form
+ Generate/Refine button and shows the list. The Save button is
in the form footer (visible only in form tabs). Switching to
History auto-loads the list (refreshed on each open so freshly-
saved entries from other tabs are visible).

**Decision: list item layout (per user spec):**
- First 2–3 lines of intent (`line-clamp-2`) — preview only,
  full text fetched on restore.
- Badges: mode (Generate/Refine, accent-colored), focus
  (`system`/`user`/`both`), example count (`"N examples"`),
  `with template` badge (only when `template_content` is non-null).
- Created timestamp (relative — `"2 hours ago"`).
- Action buttons: Restore (primary, on click anywhere on the
  card) and Delete (small icon button, top-right of the card,
  with a confirm dialog).

**Decision: API surface (new controller file
`api_prompts_history.py`):**

| Method | Path | Body / Response |
|--------|------|-----------------|
| `POST` | `/api/prompts/history` | Request: `{mode, intent, focus, examples, template_content?}`. Response: full `PromptHistoryEntry` (with `examples` list). |
| `GET` | `/api/prompts/history` | Query: `?limit=50&next=<token>`. Response: `{entries: PromptHistoryListItem[], next_token}`. List item has `intent_preview` (first 2 lines) + `example_count` + `had_template` derived server-side so the frontend doesn't have to compute them. |
| `GET` | `/api/prompts/history/<id>` | Response: full `PromptHistoryEntry`. |
| `DELETE` | `/api/prompts/history/<id>` | Response: `{status: "ok"}` or 404. |

Pagination matches the `config_history` pattern (opaque
`next` cursor = the `id` of the last entry on the current page;
server `id < before_id`).

**Phasing:**
- **Phase 7a — Backend**: migration, repository, service, controller.
- **Phase 7b — Frontend**: types, API helpers, `PromptHistoryPanel.svelte`, integration into `PromptGenerator.svelte` (third tab + Save button + restore handler).

**Files touched (planned):**
- `yadc/api/migrations/0008_prompt_history_{up,down}.sql` (new)
- `yadc/api/services/prompt_history_repository.py` (new)
- `yadc/api/services/prompt_history.py` (new)
- `yadc/api/controllers/api_prompts_history.py` (new)
- `yadc/api/services/__init__.py` (add re-exports)
- `yadc/webui/src/lib/stores/prompts/{types,api,index}.ts` (add types + API)
- `yadc/webui/src/lib/components/prompts/PromptHistoryPanel.svelte` (new)
- `yadc/webui/src/lib/components/prompts/PromptGenerator.svelte` (third tab + Save button + restore)
- `yadc/webui/src/lib/components/prompts/index.ts` (export)
- `tests/api/test_prompt_history_repository.py` (new)
- `tests/api/test_prompt_history_service.py` (new)
- `tests/api/test_api_prompts_history.py` (new)

**Doc updates:**
- This doc + implementation history entry
- `backend/api.md` — new service + controller entries
- `frontend/components-domain.md` — new `PromptHistoryPanel` entry
- `frontend/stores.md` — new history types + API functions
- `todo.md` — remove the "IndexedDB for prompt-generator few-shot examples" entry (superseded by this Phase)

**Deferred / out of scope:**
- CLI parity for save/restore — wait until Phase 4 (CLI) lands for the base flow.
- Per-entry "pin" / favorite — not requested.
- "Save over original" semantics for the refined output — cross-cutting template dialog change, already deferred from Phase 6b.
- Search/filter in the history list — list is small (≤20), not worth the UI yet.
