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

## Current tmp Commit Blocks

| Block | Position | Size | Description |
|-------|----------|------|-------------|
| A (oldest) | `[45]`, `[47-48]`, `[51]`, `[54]`, `[56]`, `[59]`, `[62]` | 8 | `tmp: frontend plan` / `tmp: update frontend plan` interleaved with docs commits |
| B | `[64]`–`[162]` | 99 | The **big block** — frontend dev (routing, datasets, captioning, toasts, SSE, tailwind, etc.) |
| C | `[164]`–`[175]` | 12 | Mixed: tmux, toast, eslint, svelte, dataset config, SSE listeners, tailwind cleanup |
| D | `[189]`–`[191]` | 3 | UI polish (vite proxy, page titles, side panel padding) |
| E (newest) | `[200]`–`[202]` | 3 | Recent memory/plan reorganizations |

## Execution Strategy

### Phase 1 — Easy wins with git-rebase-helper (Blocks E, D, C)

Squash from newest to oldest so earlier hashes remain stable.

1. **Block E** `[200]`–`[202]`
   - Squash into: `docs: reorganize agent memory files and plans`
2. **Block D** `[189]`–`[191]`
   - Squash into: `fix(webui): layout and styling polish for pages and side panel`
3. **Block C** `[164]`–`[175]` — split into 3:
   - `[164]`–`[170]` → `feat(webui): notification system and development tooling docs`
   - `[171]`–`[173]` → `feat(webui): dataset config and SSE listener fixes`
   - `[174]`–`[175]` → `style(webui): tailwind cleanup and agent memory updates`

After each squash: `list` → verify → `accept`.

### Phase 2 — The big 99-commit block (Block B)

Use `git rebase -i <parent of block>`.

- Mark all generic `tmp: save` commits → `fixup`
- Mark descriptive `tmp: …` commits → `reword` with proper `feat:`, `fix:`, `refactor:`, `docs:` messages
- Leave already-proper commits as `pick`

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
