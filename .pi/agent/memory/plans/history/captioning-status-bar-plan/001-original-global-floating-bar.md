---
date: 2025-05-28
---

# Original plan: Global floating status bar

**Context:** The original design proposed extracting captioning progress into a dedicated floating status component mounted at the layout level, visible on all pages.

**Decision:** Replaced with a simpler in-page approach (topbar spinner + absolute progress bar + side panel button swap) that requires fewer files and no layout-level changes.

**Rationale:** The global floating bar was over-engineered for the current use case. The user preferred keeping progress in the dataset browser topbar but cleaning up the layout, and using the existing side panel button for stop control.

## Original Proposal

### Problem Statement

1. **Topbar crowding** — The dataset browser topbar shows a progress bar + Stop button during captioning, competing with the back button and dataset title.
2. **Layout jump** — The subtitle row grows from ~20px (idle stats text) to ~28px (captioning row with progress bar + Stop button), causing a visible shift when captioning starts/stops.
3. **Page-locked visibility** — Progress is only visible on the dataset browser. Users browsing templates or the dataset list have no indication captioning is running.

### Proposed Solution: Global Floating Status Bar

A compact fixed-position status pill that appears whenever `captioningStatus.status` is `running` or `stopping`, regardless of the current route.

**Design:**

- **Position:** Bottom-center of viewport (above the toast stack at `z-50`)
- **Shape:** Rounded pill with semi-transparent dark background, subtle border
- **Content:**
  - Spinner icon (animated when running, stopping variant)
  - Dataset name (truncated)
  - Progress fraction (`processed/total`)
  - Thin progress bar (optional)
  - Percentage
  - **[Stop]** button (bordered, small)
  - **[×]** dismiss button (only hides the bar; it reappears on next captioning start)
- **Behavior:**
  - Auto-appears when any captioning job starts
  - Dismissable via × (sets local `hidden` flag)
  - Auto-dismisses when job reaches `done`/`error`/`idle`
  - Reappears if a new job starts while hidden
- Clicking the dataset name navigates to that dataset

### Files (original plan)

| Action | File | Purpose |
|--------|------|---------|
| Create | `CaptioningStatusBar.svelte` | Floating status component |
| Modify | `+layout.svelte` | Mount the component outside `app-shell` |
| Modify | `datasets/[name]/+page.svelte` | Remove progress bar + Stop button from topbar |

### Success Criteria (original)

- Captioning progress visible on all pages (datasets list, templates, dataset browser)
- Dataset browser topbar no longer shifts height when captioning starts/stops
- Stop button works from any page
- Status bar dismisses cleanly and reappears on next job
- No z-index conflicts with toasts or dialogs
