---
date: 2026-05-31
---
# Managed Dataset Path Guards

## Problem

Managed datasets (`source === 'upload'`) have their `[[dataset]]` paths automatically managed by the upload system (pointing to `images/` and `folders/*` inside the state directory). Users editing these paths via the config editor could:
- Break the dataset by pointing paths outside the managed directory
- Cause confusion when uploads go to unexpected locations
- Create orphaned images that are no longer tracked

## Solution

### Frontend: read-only paths

1. **Form view**: Path inputs are `disabled` for managed datasets (grayed out, non-editable)
2. **Form view**: "Add Path" button hidden for managed datasets
3. **Form view**: "Remove" button on each entry hidden for managed datasets
4. **Form view**: Extras remain fully editable
5. **Advanced view**: Small info text at the bottom warns that editing paths will error

### Backend: reject path changes

6. **`PUT /configs/<name>`**: If dataset is managed, parse the new TOML and compare `[[dataset]]` paths to the original config. Reject with 400 if they differ.
7. **`PATCH /configs/<name>`**: Same check after merging (including `dry_run`). Reject with 400 if paths differ.

Comparison is done on the raw `path` strings (set equality), so reordering is allowed but adding/removing/changing paths is not.

### Why not just frontend?

The frontend guard improves UX by making intent clear, but the backend guard is the safety net — API clients and history restores could still bypass the frontend.
