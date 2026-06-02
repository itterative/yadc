---
date: 2026-06-02
---
# Caption Section Footer Bar with Cancel + Caption/Draft Icon Buttons

**Context:** The image detail panel's Caption section had two small text-link buttons (`Caption`, `Edit`) crammed into the section header. They were hard to tap on mobile and the "Cancel" use case for single-image captioning had no UI at all. The Caption label also read `Caption → DRAFT` which read ambiguously as "caption with arrow pointing at draft" rather than "save a draft".

**Decision:**

1. **Restructured Caption section into a self-contained "caption box"** — a single `relative overflow-hidden rounded-lg bg-gray-800` container wrapping the text area on top and the action bar at the bottom. The text area (`max-h-60 overflow-y-auto`) is the only thing that scrolls; the action bar stays put.

2. **New action bar at the bottom of the caption box** — 2-column grid (`grid grid-cols-2 gap-2 p-2 bg-black/15`) that splits equally between the two halves, with `rounded-lg` buttons (`hover:bg-gray-700`, `text-sm`). Centered icon + label inside each cell. Three button sets (mutually exclusive):
   - **Viewing**: `[✨ Caption | Draft (NAME)]` (left, accent) · `[✏️ Edit]` (right, gray)
   - **Editing**: `[✕ Cancel]` (left, gray) · `[✓ Save]` (right, accent)
   - **Captioning** (single only — `isCaptioning` is only true for single-image jobs, not batch): `[✕ Cancel]` (full width, red `text-error`)
   - Truncation: span has `min-w-0 truncate`, icon has `shrink-0`, so `Draft (some-very-long-draft-name)` shows as `Draft (some-very-long-…)` when the cell is narrow

3. **Cancel button for single-image captioning** — new `handleCancelSingleCaptioning` calls the existing `stopCaptioning(datasetName)` from `captionActions`. New `isCancelling` state for visual feedback. Safe because `isCaptioning` is only true when `currentlyCaptioning` is set, which only `captionSingleImage` does (batch jobs don't set it, so the Cancel button never appears during batch).

4. **Caption button label** — `Caption → DRAFT` became `Draft (DRAFT)` to read as "save a draft named DRAFT" rather than ambiguous arrow notation. Falls back to plain `Caption` when no draft name is set.

5. **Copy button anchored to the caption box, not the scrollable text area** — moved from inside the text area div to be a direct child of the outer container with `absolute top-2 right-2 z-10`. The button now stays put while the user scrolls the caption text, so it's always reachable on long captions.

6. **New icons** — `SvgEdit` (pencil), `SvgClose` (X), and `SvgSparkle` (4-point star for Caption/generate). SvgSparkle was added in Material Symbols viewBox style (`0 -960 960 960`) to match the existing icon family.

7. **Section header simplified** — dropped the `flex items-center justify-between` wrapper since the buttons are no longer in the header; now just `<h3 class="mb-2 text-sm font-medium text-gray-300">Caption</h3>`.

**Files touched:**

- `yadc/webui/src/lib/components/dataset/ImageDetail.svelte` — imports for `stopCaptioning`, `SvgEdit`, `SvgClose`, `SvgSparkle`; new `isCancelling` state + `handleCancelSingleCaptioning`; full rewrite of the Caption section markup (caption box container, text area, copy button, action bar with 5 button states); button labels updated.
- `yadc/webui/src/lib/icons/SvgSparkle.svelte` — new icon (4-point star, Material Symbols style).
