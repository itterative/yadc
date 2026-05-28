---
name: captioning-status-bar-plan
description: Refine captioning progress UI in the dataset browser topbar and side panel button.
last_history: 1
---

# Captioning Status Bar Plan

**Status:** Phase 1 complete, Phase 2 not started

**Context:** The captioning progress bar and Stop button lived in the dataset browser topbar subtitle row. The Stop button was only accessible there, and the progress bar competed with text for horizontal space. The side panel "Start Captioning" button didn't reflect the running state.

**Goal:** Refine the existing in-page UI so the topbar is cleaner, the progress bar doesn't cause layout shifts, and the side panel button doubles as a stop control.

---

## Phase 1: Topbar + side panel button ✅

### Changes implemented

1. **Side panel button — Start → Stop swap** (`CaptionSettings.svelte`)
   - Added `isBatchCaptioning` + `onstop` props
   - Footer button shows "Stop Captioning" (error-styled `btn` with `px-4 py-2 text-sm`, `border-error/40 bg-error/20 text-error`) when running, "Start Captioning" (`btn-primary`) when idle
   - Stopping closes the panel (calls `_onclose?.()` alongside `onstop?.()`)

2. **Topbar subtitle — simplified text** (`+page.svelte`)
   - Removed inline spinner, progress bar, percentage, and Stop button
   - Running: `"Captioning… 12/50 (24%)"`
   - Stopping: `"Stopping…"`
   - Idle: unchanged stats text

3. **Spinner — right side of topbar** (`+page.svelte`)
   - Moved from inline subtitle to right of the title (`h-4 w-4`)
   - `text-accent` when running, `text-yellow-400` when stopping

4. **Progress bar — absolute under subtitle** (`+page.svelte`)
   - Thin `h-0.5` bar, `absolute -bottom-1` under the subtitle text
   - Only visible when running (not stopping)
   - `bg-accent` with smooth width transition, `bg-bg` track behind it

5. **Removed `min-h-7`** from subtitle row for consistent topbar height across pages

6. **Prop threading** (`SidePanel.svelte`)
   - Added `isBatchCaptioning` + `onstopcaptioning` props, passed through to `CaptionSettings`

### Files modified

| File | Changes |
|------|---------|
| `yadc/webui/src/routes/datasets/[name]/+page.svelte` | Restructured topbar: spinner on right of title, simplified subtitle, absolute progress bar |
| `yadc/webui/src/routes/datasets/[name]/CaptionSettings.svelte` | Added `isBatchCaptioning` + `onstop` props; footer button swaps label/action/style |
| `yadc/webui/src/routes/datasets/[name]/SidePanel.svelte` | Threaded `isBatchCaptioning` + `onstopcaptioning` props to CaptionSettings |

---

## Phase 2: Replace prop drilling with store

**Status:** Not started

**Context:** Phase 1 added `isBatchCaptioning` and `onstopcaptioning` as props threaded through `+page.svelte` → `SidePanel.svelte` → `CaptionSettings.svelte`. This is fine for two levels but is fragile — any intermediate component must pass them through. A store-based approach would let `CaptionSettings` read captioning state and call stop directly without prop drilling.

### Proposed approach

Create a thin store (or extend the existing `events.ts` store) that exposes:
- `isBatchCaptioning` — derived from `captioningStatus` + current dataset name
- `stopCaptioning()` — calls `DELETE /api/datasets/{name}/caption`

`CaptionSettings` would import and use these directly, eliminating the need for `SidePanel` to thread `isBatchCaptioning` and `onstopcaptioning` props.

### Open questions

- Should `stopCaptioning()` live in `events.ts`, `datasetImages.ts`, or a new store module?
- Should the dataset name be passed to the store function or derived from the page URL?
- Does `isBatchCaptioning` need to be dataset-aware (only true for the current dataset) or global?

### Success criteria

- [ ] `CaptionSettings` reads captioning state from store instead of props
- [ ] `CaptionSettings` calls stop via store function instead of `onstop` prop
- [ ] `SidePanel.svelte` no longer threads `isBatchCaptioning` / `onstopcaptioning`
- [ ] `+page.svelte` no longer passes those props to `SidePanel`
