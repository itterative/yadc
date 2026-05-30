---
name: git-conventions
description: Git commit message format and what to commit. Read when committing changes.
category: convention
priority: 2
---

Conventional commits format: `type: short imperative subject`

Types used: `feat:`, `fix:`, `refactor:`, `docs:`
Optional scope in parens, e.g. `refactor(cli): ...`.
Short, lowercase, imperative mood. No body/footer in recent history.

## What to commit

Always commit **all** changed files together, including:
- Source code changes
- Test files
- Agent memory files (`.pi/agent/memory/`) — these are project-level documentation and should be committed alongside the code they describe
