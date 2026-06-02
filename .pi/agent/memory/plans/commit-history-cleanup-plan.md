---
name: commit-history-cleanup-plan
description: Squash tmp commits and reword messages on feature/svelte-frontend-dataset-config-editing-improvements before merging to main.
status: In Progress
created: 2026-06-02
---

# Commit History Cleanup Plan

## Context

Branch `feature/svelte-frontend-dataset-config-editing-improvements` is 291 commits ahead of `main`. 125 of those are `tmp: …` commits that need squashing and/or rewording into conventional-commit format.

A backup already exists at `backup/pre-cleanup-20260602`.

## Goal

Reduce the branch to ~30–50 well-named conventional commits that tell a coherent story, making review and future bisection possible.

## Current tmp Commit Blocks (as of 2026-06-02)

**Phase 1 complete** — Blocks C, D, E squashed into 5 commits.

**Phase 2 in progress** — Block B (big block) being squashed in thematic chunks.

| Block | Status | Size | Description |
|-------|--------|------|-------------|
| A (oldest) | Pending | 8 | `tmp: frontend plan` / `tmp: update frontend plan` interleaved with docs commits |
| B | In Progress | ~82 remaining | The **big block** — frontend dev (routing, datasets, captioning, toasts, SSE, tailwind, etc.) |
| C | ✅ Done | 12 → 3 | `b30d859`–`7568bc7` → notification system; `ac1b30d`–`8d6fc17` → dataset config fixes; `fb4676a`–`b953ff1` → tailwind cleanup |
| D | ✅ Done | 3 → 1 | `3ffff6f`–`c77f45f` → layout/styling polish |
| E | ✅ Done | 3 → 1 | `60c3445`–`a4ca795` → memory/plan reorganization |

## Execution Strategy

### Phase 1 — Easy wins with git-rebase-helper (Blocks E, D, C)

Squash from newest to oldest so earlier hashes remain stable.

1. **Block E** `60c3445`–`a4ca795`
   - Squash into: `docs: reorganize agent memory files and plans`
2. **Block D** `3ffff6f`–`c77f45f`
   - Squash into: `fix(webui): layout and styling polish for pages and side panel`
3. **Block C** `b30d859`–`b953ff1` — split into 3:
   - `b30d859`–`7568bc7` → `feat(webui): notification system and development tooling docs`
   - `ac1b30d`–`8d6fc17` → `feat(webui): dataset config and SSE listener fixes`
   - `fb4676a`–`b953ff1` → `style(webui): tailwind cleanup and agent memory updates`

After each squash: `list` → verify → `accept`.

### Phase 2 — The big 99-commit block (Block B)

Squashed in thematic chunks using `git-rebase-helper`. Some chunks combined loosely-related commits for expediency; these may be split later if bisection or review requires finer granularity.

**Chunks squashed so far:**
- `122c8b3`–`7435266` → `feat(webui): toast notifications, captioning status, and browser alerts`
- `4112103`–`ee27acf` → `feat(webui): captioning progress UI, SSE events, templates tab, and settings` — *may split later; mixes SSE fixes, templates, progress tracking, and favicon*

Remaining chunks to squash (hashes from original list — will shift after each operation):
- `64caeaf`–`5d61393` — Mobile UI fixes and caption overlay
- `9880c09`–`040c586` — Frontend reorganization and dataset config management
- `578cde7`–`2b97e86` — CodeMirror improvements, layout, side panel
- `5f54d93`–`89179bc` — Inotify plan, implementation, and docs
- `c78fb22`–`22132bc` — Dataset thumbnails and Tailwind
- `81a481a`–`0896907` — Config/exports APIs and CodeMirror
- `276cda2`–`9c2acc8` — Editors, prompt preview, captioning progress
- `7f82d98`–`92d9ec1` — Frontend plan iterations and env endpoints
- `a6eb467`–`e1f3e88` — Initial frontend setup (routing, datasets, tailwind)

Guidelines for rewording:
- Use conventional-commit format: `type(scope): imperative subject`
- Types: `feat`, `fix`, `refactor`, `docs`, `style`, `test`
- Keep messages short; no body needed unless the change is complex
- Group closely related commits by fixup/squash

### Phase 3 — Oldest plan commits (Block A)

During the same interactive rebase (or a second one):
- Reorder the 8 `tmp: frontend plan` commits next to their neighboring `docs:` commits
- Mark them `fixup` so they disappear into the surrounding documentation commits

## Stash Warning

There is a WIP stash (`stash@{0}`) based on `d75896d`. If a full rebase from `main` is run, stash bases become orphaned. Stash should be popped or branched before Phase 2.

## Safety

- Backup branch: `backup/pre-cleanup-20260602`
- `git-rebase-helper` provides `abort` before `accept` on every operation
- After any mutating operation, run tests (`uv run pytest tests`) to catch regressions

## Done When

- `git log --oneline main..HEAD | grep -c 'tmp:'` returns 0
- All remaining commit messages follow conventional-commit format
- Branch passes `uv run pytest tests`
