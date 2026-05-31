---
date: 2025-05-30
---
# Phase 7: Refinements

**Context:** After using the implemented features, three issues emerged.

**Issues:**

### 7.1 Extras editing doesn't trigger dirty / Save button
- `KeyValueEditor` mutates `datasetEntries[i].extras` objects in-place via `bind:entries`
- `simplifiedDirty` uses `entriesEqual()` deep comparison, but Svelte's `$derived` doesn't re-evaluate because the `datasetEntries` array reference is unchanged
- Fix: make `KeyValueEditor` emit new arrays on every change (replace instead of mutate)

### 7.2 History snapshots should include current state
- Current: only saves pre-write (old) content — user can never revert TO the current version
- Missing: initial snapshot on dataset creation (import/create/upload) — no history at all for new datasets
- Fix: save post-write snapshot (the result) instead of pre-write, plus initial snapshot on creation
- Backend: move `save_snapshot` calls to after write in PUT/PATCH; add snapshot in dataset creation endpoints

### 7.3 History tab UX improvements
- Better visual hierarchy, diff view, confirmation dialog, etc.
- Details TBD from user feedback
