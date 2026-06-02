---
name: commit-history-cleanup-plan
description: Squash tmp commits and reword messages on feature/svelte-frontend-dataset-config-editing-improvements before merging to main.
status: Complete
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

**Phase 2 complete** — Block B squashed into 12 thematic commits.

**Phase 3 complete** — Block A reworded. Frontend plan commits were interleaved with implementation, so they were left as separate but properly-named `docs:` commits.

| Block | Status | Size | Description |
|-------|--------|------|-------------|
| A (oldest) | ✅ Done | 8 | `tmp: frontend plan` → `docs: frontend plan` / `docs: update frontend plan`; interleaved with implementation, left separate |
| B | ✅ Done | 99 → 12 | Squashed into thematic frontend commits |
| C | ✅ Done | 12 → 3 | Notification system, dataset config fixes, tailwind cleanup |
| D | ✅ Done | 3 → 1 | Layout/styling polish |
| E | ✅ Done | 3 → 1 | Memory/plan reorganization |

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

**Chunks squashed:**
- `122c8b3`–`7435266` → `feat(webui): toast notifications, captioning status, and browser alerts`
- `4112103`–`ee27acf` → `feat(webui): captioning progress UI, SSE events, templates tab, and settings`
- `64caeaf`–`5d61393` → `fix(webui): mobile captioning UI, editor sizing, and overlay states`
- `9880c09`–`040c586` → `refactor(webui): frontend reorganization, dataset config management, and image details`
- `578cde7`–`2b97e86` → `feat(webui): gallery layout, CodeMirror editors, side panel, and thread config`
- `5f54d93`–`89179bc` → `feat(api): inotify-based dataset watcher with plan docs and agent memory updates`
- `c78fb22`–`22132bc` → `feat(webui): dataset thumbnails, new dataset creation, and tailwind styling`
- `81a481a`–`0896907` → `feat(webui): config and export APIs, TOML editing, and CodeMirror improvements`
- `276cda2`–`9c2acc8` → `feat(webui): jinja/TOML editors, prompt preview, captioning progress, and sandbox config`
- `7f82d98`–`92d9ec1` → `feat(webui): initial frontend setup with routing, datasets, tailwind, and env endpoints`
- `a6eb467`–`e1f3e88` → `feat(api,webui): initial dataset API controllers, db migrations, and Svelte page scaffolding`

### Phase 3 — Oldest plan commits (Block A)

Reworded using `git-rebase-helper edit`:
- `tmp: frontend plan` / `tmp: update frontend plan` → `docs: frontend plan` / `docs: update frontend plan`
- Left as separate commits because they are interleaved with implementation commits
- `tmp: bash sandbox for qwen` → `docs: add bash sandbox config for qwen`
- `agent memory and instructions` → `docs: agent memory and instructions`
- `featu dataset api` → `feat: dataset api`

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
