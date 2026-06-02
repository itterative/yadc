---
date: 2026-06-02
---
# Image Detail Panel Split Into Tabbed Sub-Components

**Context:** `ImageDetail.svelte` had grown to ~480 lines containing the header (filename/image/badges/delete), the caption box (with footer bar), drafts, the unified caption+extras history, the inline TOML Extras editor, and the prompt preview — all stacked in a single scrollable column. The user wanted to edit the image's TOML sidecar directly (they use it to inject per-image prompt context like `style`/`character`/`scene`), and saw an opportunity to restructure as the panel was getting unwieldy.

**Decision:** Reorganized the panel around a `CompactPillTabs` system with three inner tabs:

- **Caption** (default, `SvgSparkle` icon) — caption box (with the existing footer bar), drafts, and the unified history. TOML editing removed from this tab.
- **Preview** (`SvgVisibility` icon) — the existing `PromptPreview`, lifted into its own tab.
- **Edit** (`SvgEdit` icon) — a new always-editable TOML editor for the image's `extras_raw` with explicit Save/Cancel and a help blurb explaining it's exposed as template variables.

Moved the file from `lib/components/dataset/ImageDetail.svelte` to `lib/components/dataset/ImageDetail/ImageDetail.svelte` and split out three sibling sub-components:

- `Caption.svelte` — caption + drafts + history rendering
- `Preview.svelte` — thin wrapper around `PromptPreview`
- `Edit.svelte` — TOML editor

Updated `SidePanel.svelte` import path.

**Architecture:** Lifted state — the parent owns `captionData`, `historyEntries`, the various loading/saving flags, `isCaptioning` (derived from `currentlyCaptioning`), `wasCaptioning` (for the re-fetch on captioning completion), `copiedKey`, and `activeTab`. Each child owns its own short-lived edit state (e.g. `Caption.svelte` owns `isEditing` / `editCaption` / `isCancelling`; `Edit.svelte` owns `editExtrasRaw` / `isDirty`). Children call back to the parent for side-effects (captioning start/cancel, saves, history restore, history lazy-load, copy) — the parent owns all the API calls and re-fetches the relevant data. The Caption child re-syncs its local `editCaption` from `captionData` via a `$effect` that respects the in-progress `isEditing` flag, so the parent's re-fetch on captioning completion flows into the textarea automatically.

**TOML history in the Edit tab:** deferred per the user's note. The backend exposes a unified image history (caption + extras per entry) — already rendered in the Caption tab's history. Duplicating it in the Edit tab would be confusing. Easy to add later as a filtered "extras-only" view if needed.

**Tradeoffs / UX notes:**

- **Cross-tab error visibility.** `captionError` (the inline error display in the caption box) is only visible on the Caption tab. The Edit tab's save failure now surfaces as a `toast.error()` instead of being silently captured into `captionError` (which the user would only see by switching back to the Caption tab).
- **History lazy-load preserved.** The "click to expand → fetch history" flow is kept; the parent owns the lazy-load logic and the child just emits `onHistoryToggle`.
- **Save flow.** `updateCaption` / `updateExtras` return `Promise<void>`, so both `handleSaveCaption` and `handleSaveExtras` re-fetch `captionData` and `historyEntries` after a successful save. The history refresh picks up the new entry that the backend just created.
- **Edit child always editable.** No Edit/Save toggle like the old inline TOML section — the tab name conveys intent. Save button disabled until `isDirty` (string compare against `captionData.extras_raw`); Cancel resets the local `editExtrasRaw` back to the saved value.

**Files touched:**

- **Added** `yadc/webui/src/lib/components/dataset/ImageDetail/Caption.svelte` (caption box + drafts + history rendering)
- **Added** `yadc/webui/src/lib/components/dataset/ImageDetail/Preview.svelte` (PromptPreview wrapper)
- **Added** `yadc/webui/src/lib/components/dataset/ImageDetail/Edit.svelte` (TOML editor)
- **Moved + rewritten** `yadc/webui/src/lib/components/dataset/ImageDetail.svelte` → `yadc/webui/src/lib/components/dataset/ImageDetail/ImageDetail.svelte` (tab system host + data layer)
- **Updated** `yadc/webui/src/routes/datasets/[name]/SidePanel.svelte` (import path)
- **Updated** `.pi/agent/memory/plans/dataset-config-ux-plan.md` (new 7.9 status item, `last_history: 34 → 35`)
- **Updated** `.pi/agent/memory/frontend-architecture.md` (new `ImageDetail/` subdirectory entry)
