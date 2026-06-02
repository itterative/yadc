---
date: 2026-06-02
---
# Image Delete Button Relocated to Filename Row

**Context:** The Delete button lived in the Caption section header of `ImageDetail.svelte`, alongside Caption and Edit. Users confused it with a caption-delete action (which doesn't exist).

**Decision:** Moved the button out of the caption area into the filename row at the top of the detail panel. Replaced the text label with the existing `SvgDelete` icon (same icon used by `DatasetManageTab.svelte` for folder deletion — consistent visual language). Icon bumped to `h-5 w-5` with `p-1.5` padding for a ~32px tap target on mobile; the row uses `flex items-center gap-2` with `<h2 class="min-w-0 flex-1 truncate">` so long filenames still truncate cleanly. Button is guarded by `source === 'upload' && item.delete_path` (unchanged). No backend or state changes — `handleDeleteImage` / `isDeleting` left as-is.

**Rationale:** Avoids adding a new kebab-menu pattern for a single secondary action. Keeps destructive actions discoverable in a header (where users look for item-level controls) while moving them out of the caption context that made the original placement ambiguous. Matches the iconography already used for folder deletion in the Manage tab.

**Files touched:**
- `yadc/webui/src/lib/components/dataset/ImageDetail.svelte` — import `SvgDelete`, restructure title row into flex with trailing icon button, remove Delete button from caption header.
