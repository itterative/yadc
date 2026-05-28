---
name: captioning-status-bar-plan
description: Refine captioning progress UI in the dataset browser topbar and side panel button.
last_history: 1
---

# Captioning Status Bar Plan

**Status:** Not started

**Context:** The captioning progress bar and Stop button currently live in the dataset browser topbar subtitle row. The Stop button is only accessible there, and the progress bar competes with text for horizontal space. The side panel "Start Captioning" button doesn't reflect the running state.

**Goal:** Refine the existing in-page UI so the topbar is cleaner, the progress bar doesn't cause layout shifts, and the side panel button doubles as a stop control.

---

## Changes

### 1. Side panel button — Start → Stop swap

**File:** `yadc/webui/src/routes/datasets/[name]/CaptionSettings.svelte`

- The footer button currently always says "Start Captioning" and calls `handleStart()`.
- When `isBatchCaptioning` is true, the button should instead say "Stop Captioning" (with a stopping-appropriate style) and call a stop callback.
- Requires passing `isBatchCaptioning` (or a derived `isCaptioning` prop) and an `onstop` callback into `CaptionSettings` from the parent page.
- When captioning is running, the rest of the form should remain visible/interactive (user may want to prepare settings for next run), but the button switches role.

### 2. Topbar subtitle — simplified text

**File:** `yadc/webui/src/routes/datasets/[name]/+page.svelte`

The subtitle row keeps its text-based status but loses the inline spinner, progress bar, and Stop button:

| State | Current | New |
|-------|---------|-----|
| Idle | `"42 images · 10 captioned · 5 with TOML"` | *(unchanged)* |
| Running | `Spinner + "Captioning… 12/50" + progress bar + "%" + Stop button` | `"Captioning… 12/50 (24%)"` |
| Stopping | `Spinner + "Stopping…"` | `"Stopping…"` |

The subtitle `min-h-7` stays to prevent layout jumps between idle/captioning states.

### 3. Spinner — right side of topbar

**File:** `yadc/webui/src/routes/datasets/[name]/+page.svelte`

- Move the spinner icon from inline in the subtitle to the **right side** of the topbar (after the title/subtitle block, at the same vertical center).
- Only visible when `isBatchCaptioning` is true.
- Uses `text-accent` color when running, `text-yellow-400` when stopping.

### 4. Progress bar — absolute under subtitle

**File:** `yadc/webui/src/routes/datasets/[name]/+page.svelte`

- Add a thin progress bar (2px height) positioned **absolutely** at the bottom of the subtitle container.
- Only visible when captioning is running (not stopping, not idle).
- Spans the full width of the subtitle container, so it visually sits under the status text.
- Uses the existing `bg-accent` color with `transition-all duration-300 ease-out` for smooth width animation.
- Since it's `position: absolute`, it does **not** increase the row height — no layout shift.

---

## Files to Modify

| File | Changes |
|------|---------|
| `yadc/webui/src/routes/datasets/[name]/+page.svelte` | Restructure topbar: move spinner to right side, add absolute progress bar under subtitle, remove inline progress bar/Stop button from subtitle text |
| `yadc/webui/src/routes/datasets/[name]/CaptionSettings.svelte` | Add `isBatchCaptioning` + `onstop` props; swap button label/action when running |
| `yadc/webui/src/routes/datasets/[name]/SidePanel.svelte` | Thread new props (`isBatchCaptioning`, `onstop`) through to `CaptionSettings` |

## Files NOT Modified

- `+layout.svelte` — no global component needed
- No new component files created

---

## Success Criteria

- [ ] Side panel button shows "Stop Captioning" when batch captioning is running, "Start Captioning" when idle
- [ ] Topbar subtitle shows status text without inline spinner/progress bar/Stop button
- [ ] Spinner appears on the right side of the topbar during captioning
- [ ] Thin absolute progress bar appears under subtitle text during captioning
- [ ] No layout shift (height change) when captioning starts or stops
- [ ] Stop button in side panel works correctly
