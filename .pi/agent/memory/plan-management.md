---
name: plan-management
description: Index of feature/design plans in plans/. Check before proposing changes that might overlap with existing or completed work.
category: meta
priority: 2
---

# Plan Management

## Storage Location

All project plans live in **`.pi/agent/memory/plans/`** inside the agent memory directory. These are **feature/design plans** — documents focused on current state: phased implementation, active decisions, and remaining work. They are separate from reference docs (architecture, conventions, workflows) which belong in `memory/docs/`.

### Directory Structure

```
plans/
├── <name>.md                  # Active plans (current state)
├── archive/<name>.md          # Completed/obsolete/superseded plans
├── archive/history/<name>/    # History of plans that have been deleted (moved from history/<name>/)
│   ├── 001-slug.md
│   └── 002-slug.md
└── history/<name>/            # Sidecar change records (per-entry files)
    ├── 001-slug.md
    └── 002-slug.md
```

Plan frontmatter may include `last_history: <N>` — the latest history entry number (omit if no history).

## Plan Index

| Plan | Description | Status |
|------|-------------|--------|
| [`dataset-config-settings-plan`](.pi/agent/memory/plans/dataset-config-settings-plan.md) | Caption settings ↔ dataset TOML config integration | Mostly implemented |
| [`dataset-config-ux-plan`](.pi/agent/memory/plans/dataset-config-ux-plan.md) | Dataset config editing UX — structured form, entries/extras, simplified/advanced toggle, revision history, TOML serialization, edit dialog rewrite with upload/manage tabs, staging conflict handling, managed dataset lifecycle | In Progress (review pass done) |
| [`cli-api-logging-unification-plan`](.pi/agent/memory/plans/cli-api-logging-unification-plan.md) | Unify the two parallel logger systems (`yadc.core.logging` and `yadc.api.modules.logging_factory`) into one. The CLI installs a `ClickHandler`; the API installs a structured `StreamHandler`; captioner code stops being a third class of code on a different logger. | Proposed |
| [`list-models-captioner-reuse-plan`](.pi/agent/memory/plans/list-models-captioner-reuse-plan.md) | Replace hand-rolled `/api/envs/<name>/models` HTTP/parsing with structured calls into the existing `APICaptioner` system. Add `list_models()` to each backend captioner, a `cmd/envs` wrapper, switch the route to GET, and add a configurable cache TTL. | In Progress |

## When to Read Plans

- User asks to implement/change/remove a feature in the index.
- User references a past design decision — also check `history/<name>/` if it exists.
- User proposes a refactor touching a planned area.
- User asks about remaining work or open questions.

## Maintenance

- **New plan** → add to index with status.
- **Plan completed** → move to `archive/`, **move `history/<name>/` to `archive/history/<name>/`**, remove from index. History travels with the plan.
- **Plan superseded** → move to `archive/`, **move `history/<name>/` to `archive/history/<name>/`**, remove from index.
- **Plan deleted** (no longer relevant, e.g. obsolete workflow no longer in use) → delete the plan file entirely; **move the matching `history/<name>/` directory to `archive/history/<name>/`** so the change log is preserved. The deleted plan's history stays accessible but is not indexed.
- **Historical content accumulated** → create numbered entry in `history/<name>/`, update `last_history` in plan frontmatter.
- **Plan details changed** → update plan file, create history entry for deviations.
- **Reference style**: `plans/<name>`, `plans/archive/<name>`, `history/<name>/<NNN>-<slug>`, `archive/history/<name>/<NNN>-<slug>`.

## History Entry Format

```markdown
---
date: YYYY-MM-DD
---
# Title
**Context:** Why. **Decision:** What. **Rationale:** Why this way.
**Files touched:** (optional)
```
