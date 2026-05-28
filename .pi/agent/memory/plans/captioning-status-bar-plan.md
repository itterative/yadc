---
name: captioning-status-bar-plan
description: Refine captioning progress UI in the dataset browser topbar and side panel button.
last_history: 2
---

# Captioning Status Bar Plan

**Status:** Complete

**Context:** The captioning progress bar and Stop button lived in the dataset browser topbar subtitle row. The Stop button was only accessible there, and the progress bar competed with text for horizontal space. The side panel "Start Captioning" button didn't reflect the running state. Caption action logic was spread across components via prop drilling.

**Goal:** Refine the topbar UI, add stop control to the side panel, and consolidate caption logic into a shared store.

---

## Phase 1: Topbar + side panel button ✅

1. **Side panel button — Start → Stop swap** (`CaptionSettings.svelte`)
   - Footer button shows "Stop Captioning" (error-styled `btn`) when running, "Start Captioning" (`btn-primary`) when idle
   - Stopping closes the panel
   - `isBatchCaptioning` derived from `captioningStatus` store, `stopCaptioning` called from store

2. **Topbar subtitle — simplified text** — Running: `"Captioning… 12/50 (24%)"`, Stopping: `"Stopping…"`, Idle: stats text

3. **Spinner — right side of topbar** — `h-4 w-4`, accent/yellow color

4. **Progress bar — absolute under subtitle** — thin `h-0.5`, `absolute -bottom-1`, only when running

5. **Removed `min-h-7`**; truncated long dataset names with `text-ellipsis text-nowrap overflow-hidden`

---

## Phase 2: Store refactor ✅

**New file: `captionActions.ts`** — shared store holding caption logic:
- `captionOptions` — writable store, reactively synced from `CaptionSettings._assembledOptions`
- `startBatchCaptioning(datasetName)` — reads options from store, `withPasswordRetry`, registers job
- `captionSingleImage(datasetName, imageId)` — same + seeds `captioningStatus` and `currentlyCaptioning`
- `stopCaptioning(datasetName, onError?)` — raw API call + toasts on failure + optional error callback
- `lastStartedJobId` — writable store for completion toast tracking

**Other changes:**
- `events.ts` — added `setCurrentlyCaptioning()` for seeding immediate spinner
- `datasetImages.ts` — `stopCaptioning` reverted to raw API call (toast logic moved to `captionActions`)
- `CaptionSettings.svelte` — writes options to store via `$effect`, calls `startBatchCaptioning` directly, removed `currentOptions`/`onstart` props
- `ImageDetail.svelte` — derives `isCaptioning` from `currentlyCaptioning` store, calls `captionSingleImage` directly, removed `isCaptioning`/`oncaptionimage`/`oncaptionupdated` props
- `SidePanel.svelte` — removed all captioning prop threading (`onstartcaptioning`, `oncaptionimage`, `oncaptionupdated`, `isCaptioning`, `captionOptions`)
- `+page.svelte` — removed `singleImageJobId`, `singleImageTargetId`, `isFocusedImageCaptioning`, `captionOptions`, `handleCaptionImage`, `_doCaptionSingle`, `handleStartCaptioning`, `_doStartCaptioning`, `handleCaptionUpdated`, single-image completion `$effect`. Uses `lastStartedJobId` store for unified completion toast.
