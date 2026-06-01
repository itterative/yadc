---
name: drop-to-upload-dataset-browser-plan
description: Drop-to-upload UX in the dataset browser page — drag files onto the image grid to open an upload modal that appends to the current managed dataset. Adds a page-level drop overlay (with warning state for non-managed / in-progress captioning) and an empty-state "Add Files" button.
status: Complete
last_history: 0
category: meta
---

# Drop-to-Upload in Dataset Browser

## Goal

Let users drop files directly onto the dataset browser page (`#/datasets/<name>`) to open an upload modal that appends the files to the current managed dataset. This removes the need to leave the page, find the dataset in the listing, click edit, then switch to the Upload tab.

## Approach

All the upload plumbing (`DatasetUploadPanel` with `mode="append"`, validation, progress, conflict resolution) already exists — it's currently nested inside `EditDatasetDialog`'s Upload tab. We just need to expose it from the dataset browser page.

### 1. Extend `DatasetUploadPanel` with `initialFiles` prop

`yadc/webui/src/lib/components/datasets/DatasetUploadPanel.svelte` — add a one-time `initialFiles?: File[]` prop that seeds the `selectedFiles` state. The user can still add more files via the FileDropZone inside the modal after it opens.

### 2. New `AddFilesDialog` component (co-located in dataset browser route)

`yadc/webui/src/routes/datasets/[name]/AddFilesDialog.svelte` — a small wrapper that mounts `Dialog` + `DatasetUploadPanel` with `mode="append"`, pre-filled `datasetName`, and an `initialFiles` prop. Reuses the same styling as `AddDatasetDialog` and `EditDatasetDialog`.

### 3. Drag-and-drop overlay on the dataset browser page

`yadc/webui/src/routes/datasets/[name]/+page.svelte` — add drag handlers on the image grid container. The overlay has three states:

- **Normal** (managed dataset, not currently captioning): accent-colored, "Drop files to add to this dataset" — drop opens the modal with the files pre-populated
- **Warning** (non-managed dataset OR currently captioning): warning-colored, explains why uploads aren't possible (e.g. "External datasets do not support file uploads" or "Captioning in progress — uploads are disabled until it finishes")
- **Hidden**: no drag in progress

Filter using `event.dataTransfer?.types.includes('Files')` so dragging text/links doesn't trigger the overlay. Counter pattern (`dragCounter`) for nested element handling.

The overlay sits on the grid container only — the side panel is not a drop target.

### 4. Empty-state "Add Files" button

Same page. The current empty state ("No images found") has no clear path forward for managed datasets. Add a prominent "Add Files" button that opens the dialog with no pre-populated files (the user can then drop or select files inside the modal).

### 5. Refresh after upload

Reuse the existing pattern from `EditDatasetDialog`: `oncomplete` callback refreshes the dataset list and reloads the image grid. The `DatasetChangedEvent` watcher will also pick up the new files automatically.

## Design decisions

- **Drop area scope**: Image grid only (not the side panel, not the topbar). Matches user intent: "drop into the browser page" = the image grid.
- **Warning overlay for blocked uploads**: Shows clear reason rather than silently failing. Reuses the same overlay infrastructure, just with different colors and message.
- **Empty state button**: Complements drag-and-drop for click/keyboard users.
- **Mobile interaction**: Deferred. The user requested deferring mobile-specific handling. Standard drop targets don't work on touch devices, but the empty-state button still provides an entry point.
- **Captioning in progress**: Backend already blocks uploads with 409. We show a warning overlay proactively (better UX than letting the user try and fail).

## Files affected

- `yadc/webui/src/lib/components/datasets/DatasetUploadPanel.svelte` — add `initialFiles` prop
- `yadc/webui/src/routes/datasets/[name]/AddFilesDialog.svelte` (new) — wrapper
- `yadc/webui/src/routes/datasets/[name]/+page.svelte` — drag/drop overlay + dialog + empty-state button

## Out of scope

- Mobile touch interaction for drag-and-drop (deferred)
- Topbar "Add Files" button (the empty-state button + EditDatasetDialog from the listing cover click-only entry points)
- Refactoring `FileDropZone` to extract the `webkitGetAsEntry` file collection logic (not needed — duplicating ~30 lines is acceptable for now)
