---
name: plan-management
description: When a user query relates to features, refactors, or design work that may have an existing plan, always read the relevant plan(s) first before proposing changes.
---

# Plan Management

## Storage Location

All project plans live in **`.pi/agent/memory/plans/`** inside the agent memory directory. These are **feature/design plans** — documents with phased implementation, checklists, architecture decisions, and historical context. They are separate from reference docs (architecture, conventions, workflows) which belong in `memory/docs/`.

## Plan Index

| Plan | Description | Status |
|------|-------------|--------|
| [`dataset-config-settings-plan`](.pi/agent/memory/plans/dataset-config-settings-plan.md) | Integrating caption settings panel with dataset TOML config. Covers `PATCH /configs/<name>`, typed Config API (Pydantic + TS), diff indicators, overrides section, "Save as dataset default". Remaining: preset profiles, config diff banner, TOML multiline string serialization for templates. | Mostly implemented |
| [`file-based-private-key-plan`](.pi/agent/memory/plans/file-based-private-key-plan.md) | Complete historical summary of replacing `keyring`-based RSA private key storage with a password-protected alternative (PBKDF2 + AES-256-GCM in config TOML). Tracks original intent, deviations, decisions, and all files touched. | Complete |
| [`frontend-plans`](.pi/agent/memory/plans/frontend-plans.md) | Frontend implementation status, backend controller endpoint index, component design notes (CodeMirror, editors, selectors), and known issues. All phases 1–4 implemented. | Complete |

## When to Review Plans

**Before any of the following, read the relevant plan(s):**
- The user asks to implement, change, or remove a feature that appears in the index above.
- The user references something that sounds like a past design decision (e.g. "why do we have two key pairs?", "how does config patching work?").
- The user proposes a refactor touching areas covered by a plan.
- The user asks about remaining work, TODO items, or open questions from a past effort.

## Keeping Plans Up to Date

- **When a plan is completed**, update its status in this index to "Complete".
- **When a plan has remaining items**, keep its status accurate (e.g. "Mostly implemented", "In progress").
- **When a new plan is created**, add it to the index with a concise description and status.
- **When a plan is fully superseded or obsolete**, move it to a `plans/archive/` subdirectory (create if needed) and remove it from the active index.
- **When a plan's details change**, update the plan file itself, not just this index. Cross-reference `todo.md` if remaining work is tracked there instead.
- **Reference style**: use `plans/<name>` in other memories and `todo.md` so links remain valid if the directory structure changes.
