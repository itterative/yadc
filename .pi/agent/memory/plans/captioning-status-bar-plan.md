---
name: captioning-status-bar-plan
description: Refine captioning progress UI in the dataset browser topbar and side panel button.
last_history: 2
---

# Captioning Status Bar Plan

**Status:** Phase 1 complete, Phase 2 not started

**Context:** The captioning progress bar and Stop button lived in the dataset browser topbar subtitle row. The Stop button was only accessible there, and the progress bar competed with text for horizontal space. The side panel "Start Captioning" button didn't reflect the running state.

**Goal:** Refine the existing in-page UI so the topbar is cleaner, the progress bar doesn't cause layout shifts, and the side panel button doubles as a stop control. Then eliminate prop drilling by moving caption logic into a shared store.

---

## Phase 1: Topbar + side panel button ✅

### Changes implemented

1. **Side panel button — Start → Stop swap** (`CaptionSettings.svelte`)
   - Footer button shows "Stop Captioning" (error-styled `btn`) when running, "Start Captioning" (`btn-primary`) when idle
   - Stopping closes the panel
   - `isBatchCaptioning` derived from `captioningStatus` store, `stopCaptioning` called from store

2. **Topbar subtitle — simplified text** (`+page.svelte`)
   - Running: `"Captioning… 12/50 (24%)"`
   - Stopping: `"Stopping…"`
   - Idle: unchanged stats text

3. **Spinner — right side of topbar** (`+page.svelte`)
   - `h-4 w-4`, `text-accent` when running, `text-yellow-400` when stopping

4. **Progress bar — absolute under subtitle** (`+page.svelte`)
   - Thin `h-0.5` bar, `absolute -bottom-1`, only when running

5. **Removed `min-h-7`** from subtitle row; truncated long dataset names with `text-ellipsis text-nowrap overflow-hidden`

### Files modified

| File | Changes |
|------|---------|
| `yadc/webui/src/routes/datasets/[name]/+page.svelte` | Restructured topbar, removed `handleStopCaptioning` and unused imports |
| `yadc/webui/src/routes/datasets/[name]/CaptionSettings.svelte` | Derives `isBatchCaptioning` from store, calls `stopCaptioning` directly, removed `isBatchCaptioning`/`onstop` props |
| `yadc/webui/src/routes/datasets/[name]/SidePanel.svelte` | Removed `isBatchCaptioning`/`onstopcaptioning` prop threading |

---

## Phase 2: Store refactor — eliminate prop drilling

**Status:** Not started

**Goal:** Move all caption action logic (start batch, start single, stop, options) into a shared store so components can act directly without prop chains.

### New file: `captionActions.ts`

```
yadc/webui/src/lib/stores/captionActions.ts
```

Exposes:
- **`captionOptions`** — writable store holding the current assembled `CaptionOptions`. Written by `CaptionSettings`, read by action functions.
- **`startBatchCaptioning(datasetName)`** — reads `captionOptions` from store, calls `withPasswordRetry(() => startCaptioning(...))`, registers job ID, seeds `captioningStatus` store. Returns job ID for caller tracking.
- **`captionSingleImage(datasetName, imageId)`** — reads `captionOptions` from store, calls `withPasswordRetry(() => captionSingleImage(...))`, registers job ID, seeds `captioningStatus` + `currentlyCaptioning` stores.
- **`stopCaptioning(datasetName, onError?)`** — moved from `datasetImages.ts`. Shows toast on failure, optional error callback.

### Changes per file

#### `CaptionSettings.svelte`
- Remove `onstart`, `onclose` props
- `handleStart()` writes assembled options to `captionOptions` store (instead of `currentOptions` bindable + `onstart` callback), then calls `captionActions.startBatchCaptioning(datasetName)`
- Remove `currentOptions` bindable prop
- `onclose` — call directly or keep as prop (for closing the panel drawer)

#### `ImageDetail.svelte`
- Remove `isCaptioning`, `oncaptionimage`, `oncaptionupdated` props
- Import `currentlyCaptioning` from `events.ts`, derive `isCaptioning` locally:
  ```
  isCaptioning = currentlyCaptioning?.dataset_name === datasetName && currentlyCaptioning?.image_id === item.id
  ```
- "Caption" button calls `captionActions.captionSingleImage(datasetName, item.id)` directly
- Remove `wasCaptioning` completion effect (caption reload still happens via the existing load-on-item-change effect, or we keep a simplified version)

#### `SidePanel.svelte`
- Remove `onstartcaptioning`, `oncaptionimage`, `oncaptionupdated`, `isCaptioning`, `captionOptions` props
- Much thinner — just layout and tab routing

#### `+page.svelte`
- Remove: `captionOptions` state, `singleImageJobId`, `singleImageTargetId`, `isFocusedImageCaptioning`, `_doCaptionSingle`, `handleCaptionImage`, `_doStartCaptioning`, `handleStartCaptioning`, `handleCaptionUpdated`, single-image completion `$effect`
- Keep: `runningJobId` + batch completion `$effect` (for unified toast), `isBatchCaptioning` (for topbar), SSE tile update effects
- Simplified `<SidePanel>` call with fewer props

#### `datasetImages.ts`
- `stopCaptioning()` moves to `captionActions.ts`

### Success criteria

- [ ] New `captionActions.ts` store with `captionOptions`, `startBatchCaptioning`, `captionSingleImage`, `stopCaptioning`
- [ ] `CaptionSettings` writes options to store and calls `startBatchCaptioning` directly
- [ ] `ImageDetail` derives `isCaptioning` from `currentlyCaptioning` store
- [ ] `ImageDetail` calls `captionSingleImage` directly
- [ ] `SidePanel` has no captioning-related prop threading
- [ ] `+page.svelte` has no single-image captioning state
- [ ] `oncaptionupdated` removed (SSE effects handle tile updates)
- [ ] Unified batch completion toast still works
