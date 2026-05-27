---
name: captioning-status-bar-plan
description: Move captioning progress UI out of the dataset browser topbar into a global floating status bar.
---

# Captioning Status Bar Plan

**Status:** Not started

**Context:** The captioning progress bar and Stop button currently live inside the dataset browser topbar (`datasets/[name]/+page.svelte`). This crowds the topbar and makes the subtitle row taller than the idle stats line, causing layout shifts. It also means progress is only visible on the dataset browser page.

**Goal:** Extract captioning progress into a dedicated floating status component mounted at the layout level, visible on all pages.

---

## Problem Statement

1. **Topbar crowding** — The dataset browser topbar shows a progress bar + Stop button during captioning, competing with the back button and dataset title.
2. **Layout jump** — The subtitle row grows from ~20px (idle stats text) to ~28px (captioning row with progress bar + Stop button), causing a visible shift when captioning starts/stops.
3. **Page-locked visibility** — Progress is only visible on the dataset browser. Users browsing templates or the dataset list have no indication captioning is running.

---

## Proposed Solution: Global Floating Status Bar

A compact fixed-position status pill that appears whenever `captioningStatus.status` is `running` or `stopping`, regardless of the current route.

### Design

- **Position:** Bottom-center of viewport (above the toast stack at `z-50`, so this component should use `z-50` too, or the stack should be `z-50` and this `z-50` as well — they won't overlap).
- **Shape:** Rounded pill with semi-transparent dark background, subtle border.
- **Content:**
  - Spinner icon (animated when running, stopping variant)
  - Dataset name (truncated)
  - Progress fraction (`processed/total`)
  - Thin progress bar (optional, can be omitted for minimalism)
  - Percentage
  - **[Stop]** button (bordered, small)
  - **[×]** dismiss button (only hides the bar; it reappears on next captioning start)
- **Behavior:**
  - Auto-appears when any captioning job starts
  - Dismissable via × (sets local `hidden` flag)
  - Auto-dismisses when job reaches `done`/`error`/`idle`
  - Reappears if a new job starts while hidden

### Files to Create

| File | Purpose |
|------|---------|
| `yadc/webui/src/lib/components/ui/CaptioningStatusBar.svelte` | Floating status component |

### Files to Modify

| File | Changes |
|------|---------|
| `yadc/webui/src/routes/+layout.svelte` | Mount `<CaptioningStatusBar>` outside `app-shell` |
| `yadc/webui/src/routes/datasets/[name]/+page.svelte` | Remove progress bar + Stop button from topbar; keep idle stats line |
| `yadc/webui/src/routes/datasets/[name]/+page.svelte` | Remove `handleStopCaptioning` (or move it to the status bar component) |

### API

The component uses the existing `captioningStatus` store (`$lib/stores/events`) and calls `DELETE /api/datasets/{name}/caption` for the Stop button.

### Open Questions

- Should the bar show for **all** datasets or only the one on the current page? (Show globally — it's a shared resource.)
- Should clicking the dataset name navigate to that dataset? (Yes, useful when on another page.)
- Should the progress bar be included or is fraction + % enough? (Start with fraction + % for compactness; add bar if it feels too minimal.)

---

## Success Criteria

- [ ] Captioning progress visible on all pages (datasets list, templates, dataset browser)
- [ ] Dataset browser topbar no longer shifts height when captioning starts/stops
- [ ] Stop button works from any page
- [ ] Status bar dismisses cleanly and reappears on next job
- [ ] No z-index conflicts with toasts or dialogs

---

## Related Plans

- `frontend-plans` — Topbar padding / height inconsistency (will be resolved by this plan)
