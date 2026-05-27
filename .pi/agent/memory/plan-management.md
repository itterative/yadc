---
name: plan-management
description: When a user query relates to features, refactors, or design work that may have an existing plan, always read the relevant plan(s) first before proposing changes.
---

# Plan Management

## Storage Location

All project plans live in **`.pi/agent/memory/plans/`** inside the agent memory directory. These are **feature/design plans** — documents focused on current state: phased implementation, active decisions, and remaining work. They are separate from reference docs (architecture, conventions, workflows) which belong in `memory/docs/`.

### Directory Structure

```
plans/
├── <name>.md                  # Active/completed plans (current state)
├── archive/<name>.md          # Obsolete/superseded plans
└── history/<name>/            # Sidecar change records (per-entry files)
    ├── 001-slug.md
    └── 002-slug.md
```

Plan frontmatter may include `last_history: <N>` — the latest history entry number (omit if no history).

## Plan Index

| Plan | Description | Status |
|------|-------------|--------|
| [`async-migration-plan`](.pi/agent/memory/plans/async-migration-plan.md) | Flask/waitress → Quart/uvicorn async migration | Complete |
| [`dataset-config-settings-plan`](.pi/agent/memory/plans/dataset-config-settings-plan.md) | Caption settings ↔ dataset TOML config integration | Mostly implemented |
| [`file-based-private-key-plan`](.pi/agent/memory/plans/file-based-private-key-plan.md) | Replace keyring with password-encrypted config TOML storage | Complete |
| [`frontend-plans`](.pi/agent/memory/plans/frontend-plans.md) | Frontend implementation status, component index, design notes | Complete |
| [`image-upload-dataset-creation-plan`](.pi/agent/memory/plans/image-upload-dataset-creation-plan.md) | Upload images/folders from browser when creating a dataset via WebUI | Not started |

## When to Read Plans

- User asks to implement/change/remove a feature in the index.
- User references a past design decision — also check `history/<name>/` if it exists.
- User proposes a refactor touching a planned area.
- User asks about remaining work or open questions.

## Maintenance

- **New plan** → add to index with status.
- **Plan completed** → set status to "Complete".
- **Plan superseded** → move to `archive/`, remove from index; history stays in `history/`.
- **Historical content accumulated** → create numbered entry in `history/<name>/`, update `last_history` in plan frontmatter.
- **Plan details changed** → update plan file, create history entry for deviations.
- **Reference style**: `plans/<name>`, `history/<name>/<NNN>-<slug>`.

## History Entry Format

```markdown
---
date: YYYY-MM-DD
---
# Title
**Context:** Why. **Decision:** What. **Rationale:** Why this way.
**Files touched:** (optional)
```
