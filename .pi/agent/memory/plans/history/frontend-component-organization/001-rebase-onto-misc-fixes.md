---
date: 2026-06-03
---
# Rebase onto fix/svelte-frontend-misc-fixes

**Context:** The user requested a rebase onto `fix/svelte-frontend-misc-fixes` before executing the plan, because that branch had landed changes that might affect it.

**Decision:** Stash + rebase + pop. Rebase succeeded with one auto-merge on `todo.md`. The branch brought in three frontend commits:

- `8374ee64 feat(webui): add Card, ActionBar, ActionBarItem components` — three new `ui/` primitives used by `Caption.svelte`, `Extras.svelte`, `ConfigHistory.svelte`, `EditDatasetDialog.svelte`, `SidePanel.svelte`.
- `62c33673 feat(webui): add delete and promote actions for history and drafts in image details` — `Caption.svelte` grew from 381 to 409 lines, `ImageDetail.svelte` from 359 to 376 lines, new `SvgUpgrade.svelte` icon.
- `f06378ef feat(webui): add draft management to dataset settings manage tab` — `DatasetManageTab.svelte` grew from 101 to 183 lines.

(The fourth commit `ef96fdc1 fix: switch DatasetImage serialization from toml to tomlkit` was backend-only.)

**Rationale:**

- The new `ui/` primitives align perfectly with the locked-in "atomic primitive in `ui/`" rule. Added to the target structure's `ui/` line.
- `Caption.svelte` is still 409 lines but the new `Card`/`ActionBar`/`ActionBarItem` structure made it much more readable (clearly delineated sections of `<Card>...</Card>`). Demoted the planned `Caption.svelte` split from recommended to optional.
- `DatasetManageTab.svelte` at 183 lines still fits a single file; no structural change needed beyond the move into `dataset/manage/`.
- No conflicting code paths — all the rebase changes are additions, no renames or behavior changes that affect the plan.

**Plan impact:** Target structure updated with the new `ui/` primitives and post-rebase line counts. `Caption.svelte` split demoted. No structural changes to the plan's other phases.
