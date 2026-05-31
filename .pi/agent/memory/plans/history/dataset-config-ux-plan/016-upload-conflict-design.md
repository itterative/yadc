---
date: 2026-05-31
---
# Upload Conflict Handling — Design Decision

**Problem:** When appending files to an existing managed dataset, files with the same name as existing ones are silently overwritten. This is risky and gives the user no control.

## Options Considered

### A: Reject conflicts (all-or-nothing)
Check for existing files before writing anything. If any conflict exists, abort the entire upload and return an error listing the conflicts.

- **Pros:** Trivial to implement, zero risk of data loss, no staging I/O.
- **Cons:** One conflict blocks the whole batch; user must manually juggle source files.

### B: Dry-run preflight (two-phase)
Add `?dry_run=true` to the append endpoint. It validates and scans for conflicts but doesn't write. Frontend calls dry-run first, shows confirmation if conflicts exist, then calls real upload with `?confirm_overwrite=true`.

- **Pros:** Good UX, no staging I/O overhead, user sees what will happen.
- **Cons:** Race condition between dry-run and upload; doesn't support "keep both" / rename.

### C: Lightweight staging (selected)
Write uploads to a **staging directory** first, then do conflict resolution before committing to the live dataset.

- **Pros:** No partial overwrites; supports per-file resolution (skip / overwrite / rename); rich preview possible; extensible.
- **Cons:** More backend + frontend work.

## Selected Design: Staging Folder with Periodic Cleanup

**User choice:** Option C — staging folder per dataset, with periodic cleanup.

### Backend Design

**Staging directory layout:**
```
STATE_PATH/<name>/.staging/<upload-id>/
  images/
  folders/
```

- Each upload gets a unique subfolder under `.staging/` (e.g. UUID or timestamp-based)
- `append_dataset_from_upload()` writes to the staging folder instead of live dirs
- After staging completes, emit a `phase: "conflicts"` event listing collisions (filename, existing size, new size)
- New endpoint: `POST /datasets/<name>/staging/commit` with body:
  ```json
  {
    "staging_id": "<upload-id>",
    "resolutions": {
      "image.jpg": "overwrite",
      "other.jpg": "skip"
    }
  }
  ```
- Commit endpoint applies resolutions, moves files to live dirs, appends new `[[dataset]]` entries if needed, rescans, deletes the staging subfolder
- Before each new append upload, wipe any leftover `.staging/<upload-id>` dirs from prior failed/aborted uploads

**Periodic cleanup:**
- A background thread scans all `.staging/` subfolders across all datasets
- If a staging folder's creation/modified timestamp is older than a threshold (e.g. 24h), delete it
- This handles uploads that were staged but never committed (user closed browser, backend crash, etc.)

**Open question:** Should staging live in `STATE_PATH/<name>/.staging/` or a global cache directory? Using the state folder keeps staging co-located with the dataset data (simpler atomic moves, same filesystem). A cache folder would be more conventional for temp data but complicates cross-filesystem moves.

### Frontend Design

1. Upload to staging as usual — progress bar shows "Staging..."
2. If no conflicts: auto-commit and close
3. If `conflicts` event arrives: freeze dialog, show conflict resolution table
   - Per-file action dropdown: Skip / Overwrite / Keep Both (auto-rename)
   - Bulk actions: "Skip All", "Overwrite All"
   - Optional: show file sizes for context
4. User clicks "Apply", frontend calls commit endpoint
5. Show "Applying..." spinner, then close dialog and refresh listing

### Files to Touch (future implementation)

- `yadc/api/services/dataset_upload.py` — staging write, conflict detection, commit logic
- `yadc/api/controllers/api_datasets.py` — `POST /datasets/<name>/upload` (staging mode), `POST /datasets/<name>/staging/commit`
- `yadc/api/modules/dataset_watcher.py` or new cleanup module — periodic stale staging cleanup
- `yadc/webui/src/lib/components/datasets/DatasetUploadPanel.svelte` — handle conflicts event, show resolution UI
- `yadc/webui/src/lib/stores/datasetImages.ts` — `commitStagingUpload()` helper
