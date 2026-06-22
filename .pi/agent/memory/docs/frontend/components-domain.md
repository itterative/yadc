---
name: frontend/components-domain
description: Frontend lib/components/<domain>/ — dataset, caption, env, export, settings, dialogs, icons. Domain components (used in ≥2 places OR used by a lib/ component).
category: architecture
---

# Frontend: `lib/components/<domain>/`

Domain components. Per the "scope rule" (in `frontend-architecture`), these are components used in ≥2 places OR used by a `lib/` component. Each domain follows the "feature-folder rule" for sub-folders with ≥3 related files (host + tabs/sections).

## See also (in `.pi/agent/memory/docs/`)

- `frontend-patterns` — Drop-to-upload (DropUploadZone + AddFilesDialog)
- `dataset-system` — Upload pipeline, Dataset creation flows, DatasetImage persistence

## dataset/

```
yadc/webui/src/lib/components/dataset/
  browser/                      # Image grid + tile
    DatasetBrowser.svelte       # Masonry grid container (column distribution + infinite scroll + selectedId)
    DatasetImage.svelte         # Masonry grid tile (thumbnail + badges + selected outline)
  detail/                       # Image detail (tab host + tabs)
    ImageDetail.svelte          # Tab system host (CompactPillTabs: Caption / Preview / Edit) + data layer (captionData / historyEntries / API calls). History auto-loaded alongside caption.
    Caption.svelte              # Caption box (ActionCard + ActionBar) + Drafts (Card+ActionBar: Promote/Delete) + History (always visible when entries exist, Restore/Delete by content hash)
    RefineDialog.svelte         # Dialog for caption refinement — editable caption + feedback input, sends to model as conversation, shows result with Accept/Retry
    Preview.svelte              # Inlines the caption/template preview markup directly (the old `ui/PromptPreview.svelte` wrapper was deleted; the freed `PromptPreview` name was reused by the unrelated `prompts/GenerationPreview`).
    Extras.svelte               # Always-editable TOML editor for the image's extras_raw (ActionCard + ActionBar with Save/Cancel)
  config/                       # Dataset config editor (tab host + tabs + helpers)
    DatasetConfig.svelte        # Tab system host (Form / Advanced / History) + data fetching + save orchestration
    DatasetConfigForm.svelte    # Structured form view of the config — fields bound to `configState` Svelte 5 `$state` rune from `state.svelte.ts`
    DatasetConfigAdvanced.svelte # Raw TOML editor view of the config. Binds to `configState.rawContent`; shows a warning when the structured form has unsaved changes.
    ConfigApiSection.svelte     # API section of the form: env dropdown + URL/model fields. Placeholders update from the selected env; model field is a select when models are loaded, input otherwise; refresh button. Binds to `configState` (env-aware, but does NOT pre-fill values like `EnvSelector` does for the caption flow).
    ConfigHistory.svelte        # Config revision history browser (list, view, restore entries). Uses ActionCard + ActionBar per entry. Reads from `ConfigHistoryRepository` on the backend.
    patch.ts                    # Pure TOML patch helpers (e.g. `assemblePatchFromForm`) — no Svelte/DOM dependencies, testable in isolation.
    state.svelte.ts             # Svelte 5 `$state` rune module holding the form / raw content state shared between `DatasetConfigForm` and `DatasetConfigAdvanced`. Exposes `simplifiedDirty` flag.
  upload/                       # Dataset upload (panel + slim tab wrappers)
    DatasetUploadPanel.svelte   # Shared upload panel (create + append modes). When `mode="append"`, accepts an `initialFiles` prop (consumed once via `untrack()`) to pre-populate the file list from a drop event on the page.
    UploadDatasetTab.svelte     # Upload tab — file selection, progress bar, cancel upload, toast warnings
    CreateDatasetTab.svelte     # Create/Import tab — radio toggle between "Add image paths" and "Import TOML config"
  manage/                       # Dataset manage (delete + folder ops)
    DatasetManageTab.svelte     # "Manage" tab inside the dataset creation dialog — list managed (upload-sourced) datasets with folder/image counts and delete actions.
```

## caption/

```
  caption/                        # Caption domain
    CaptionSettingsPanel.svelte   # The batch-captioning side panel. Derives isBatchCaptioning from store, shows Start/Stop button. Writes assembled options to captionActions store reactively. Fetches dataset config defaults, shows diff dots for overrides. Auto-saves overrides to `captionSettings` localStorage via `deferred()` on any field change; flushes pending save before starting captioning.
    OverridesSection.svelte       # Sub-section of the captioning panel: per-field reset/override UI for the localStorage-saved overrides
    TemplateSection.svelte        # Sub-section of the captioning panel: template dropdown + readonly `JinjaEditor` card with Edit/Delete action bar. Edit and `+` button open `EditTemplateDialog` (reused from `lib/components/templates/`); Delete uses `confirmDialog.danger(...)`. The "create new template" flow is dialog-based — no in-place editing of template content.
```

## env/

```
  env/                            # Env domain
    EnvironmentSettings.svelte    # Environment CRUD tab (env list, inline edit, token reveal)
    EnvSelector.svelte            # Environment form (env dropdown + URL/token/model, bindable props). "Manage…" link opens SettingsDialog at the Environments tab.
```

## templates/

```
  templates/                      # Templates domain
    EditTemplateDialog.svelte     # Reusable create/edit template dialog. `templateName=null` for new; otherwise edits the named template. Optional `ephemeral` mode (used by `prompts/PromptForm` for one-off refine templates) returns the editor content via `onapply` instead of saving to the backend — hides the name field, rebrands Save → Apply. Used by the `/templates` route, `caption/TemplateSection`, and the prompt generator (Save-as-template + refine-ephemeral). `initialContent?` pre-fills the editor for new/ephemeral templates.
    index.ts                      # Barrel re-export (`$lib/components/templates`)
```

## prompts/

```
  prompts/                        # Prompt-generator domain — feature folder
    PromptGenerator.svelte        # Host for the prompts route. Layout: full-page `GenerationPreview` on the left + `PromptSidePanel` (form/settings/history tabs; the Generate tab also carries the Save-to-history + Generate footer) on the right. Owns the form-state `$state`s (mode, env, apiUrl, apiToken, apiModelName, intent, focus, maxTokens, imageQuality, templateContent, examples) and the persistence `$effect` that mirrors them into `promptSettings` on every change. Also owns the `EditTemplateDialog` flow — the preview's "Save as template" button opens this dialog (it's not triggered from the side panel). `ongenerate` callback (passed to the panel) calls `startGeneration` with a snapshot of form values so mid-stream edits don't leak into the in-flight call. Re-derives `canGenerate` locally as a defense-in-depth guard for `handleGenerate` (the panel derives its own copy to drive its button).
    PromptSidePanel.svelte        # Tabbed side panel hosting the three settings tabs (Generate / Settings / History). The Generate tab carries an `ActionBar` footer (Save-to-history secondary + Generate/Refine primary); during streaming the primary swaps to Cancel (danger) and Save is hidden — a single unambiguous stop action. Built on the generic `ui/SidePanel` primitive — desktop inline (`lg:static lg:w-120`), mobile slide-over drawer whose `fab` snippet swaps the FAB between a streaming cancel and the default toggle (via `FabButton`). The FAB mirrors the footer's Cancel so the user can stop the in-flight call whether the panel is open or closed. Form state is bound through from the host (`bind:mode`, `bind:intent`, etc.); `open` and `activeTab` are also bindable so the mobile X (in the `PillTabs` `end` snippet) can close the drawer and so the panel's restore handler can switch back to the Generate tab. Owns the `promptForm` ref (for `restoreTemplate`) + `historyPanel` ref + `isSaving` state + `handleSaveToHistory` + `handleRestore`. Reads `generation.status` directly from the store to derive `isStreaming` / `canGenerate` / `canSave` for the footer buttons. Refine mode no longer requires a template (none set → the backend runs its generate branch and emits a new template), so `canGenerate`/`canSave` are gated only by env + intent.
    PromptForm.svelte             # The Generate-tab form: intent textarea, focus toggle, mode toggle (New/Refine pills, bound to host `mode` via `bind:mode`), template section (refine only), few-shot examples section. Selecting Refine reveals the Template section — a picker `<select>` over `$templates.items` (with a `(custom)` empty option) + a `Card` (mirrors `caption/TemplateSection`) whose body is a fixed-height `JinjaEditor` and whose `ActionBar` has a single Edit button — and relabels the intent field ("Refinement intent"). The card is **never inline-editable**: an existing template shows read-only + Edit (opens `EditTemplateDialog`, saves back to the backend); the `(custom)` option shows either a read-only editor of custom content or a placeholder ("No template set. Refining will generate a new template…"), and Edit opens `EditTemplateDialog` in **ephemeral mode** (`onapply` returns the content, not persisted). Custom edits are held in an internal `customTemplateContent` state (NOT the bindable `templateContent`) so switching to an existing template and back preserves them; `templateContent` (bound to host) is always the *effective* content — restored from `customTemplateContent` when the picker is on `(custom)` (the load effect's `!name` branch), set to the loaded template when a named one is picked. After a backend-edit save, `handleEditSaved` toggles `selectedTemplateName` ('' → name) to force a refetch. Variables chip strip below the card. Env/model/generation-limits live in `PromptSettings` (the Settings tab), NOT here. Rendered once by the host, so internal state (template picker selection, loading flags) survives a mode switch.
    PromptSettings.svelte         # The Settings-tab form: Environment section (env `<select>` + API URL + API token + model picker with refresh, auto pre-filled from the selected env) moved out of `PromptForm`, plus a Generation-limits section — `max_tokens` (number input, default 16384; the client default of 4096 truncates longer templates) and `image_quality` (auto/low/high pills, few-shot example fidelity → OpenAI `image_url.detail` / Gemini `mediaResolution`). All fields bindable against the host's persisted `promptSettings`; disables its inputs while a generation is streaming. Mirrors `caption/EnvSelector` for the env/model part.
    ExamplesPanel.svelte          # Few-shot examples as a read-only card list (thumbnail + subject + caption, truncated) with hover-revealed corner Edit/Delete icon buttons (templates-page pattern — always visible on touch via `max-lg:opacity-100`). Editing or adding goes through dialogs — the card itself is never inline-editable. A single bottom `ActionBar` (Add manual secondary / From dataset secondary) is the one add entry point (shown in both the empty and non-empty states). Manual add = hidden single-file `<input>` → `readFileAsDataUrl` → `EditExampleDialog` seeded with subject=file_stem, caption="" → appends on save. From dataset → `DatasetPickerDialog` (multi-select → batch append). Delete is direct (no confirm — examples aren't persisted). Hosts both dialogs; owns the edit state (`editIndex` null=new vs index=update, plus the seeded image/subject/caption). `disabled` (streaming) threads to the icons + ActionBar. `PromptForm` renders this under the "Few-shot examples" heading + description.
    EditExampleDialog.svelte      # Dialog editing ONE example's subject + caption with a full-bleed **hero banner** image (RefineDialog pattern: `-mx-5` breakout, `h-32 object-cover`) right under the header. The banner crops (object-cover), so it's clickable (cursor-zoom-in) → opens the shared `ui/ImagePreviewDialog` lightbox (`object-contain`, fills most of the screen, handles wide AND tall images via `max-h-[inherit] w-full flex-1 object-contain` + optional `style:aspect-ratio`). A non-interactive `SvgFullscreen` chip (bottom-right, `bg-black/60`, `pointer-events-none`) signals the enlarge affordance — the whole banner is the click target. Native modal dialogs stack in the top layer, so Escape closes the lightbox first (not the edit dialog) and the backdrop dims the edit dialog behind. Used for both add-manual (ExamplesPanel seeds it from the picked file) and edit-existing. Mirrors `EditTemplateDialog`'s shell (`Dialog` + header + body + `btn-bar`). Props: `open`, `title` ("Add example" vs "Edit example"), `imageDataUrl`, `initialSubject`, `initialCaption`, `onsave({subject, caption})`, `onclose`. Resets from the seed values on each open (mirrors EditTemplateDialog's reset-on-open). The image is NOT replaceable here — change it by deleting + re-adding.
    DatasetPickerDialog.svelte    # Dialog for batch-adding examples from a dataset: dataset `<select>` + multi-select image grid (click toggles selection via a `SvelteSet<number>` of image ids; selected tiles get `border-accent` + an accent checkmark chip). Footer shows "N selected" + Cancel + "Add N" (disabled at 0). On Add: sequentially `fetchImageAsDataUrl` (full-res media — a few-shot example needs the real image) + best-effort `fetchCaption` per selected image, building `ExamplePair`s (subject=file_stem, caption=fetched-or-empty), returned via `onadd` and appended. Sequential (not parallel) to avoid hammering the server with N concurrent media fetches; a progress label ("Adding 2 of 5…") tracks it. Selection clears on dataset switch (image ids are per-dataset, so stale ids would match the wrong images). Uses a component-level abort context + per-open `linkedController`s so closing the dialog cancels in-flight dataset/image loads.
    GenerationPreview.svelte      # Streaming monospace preview (full-bleed — no card chrome or title on the panel itself) with blinking caret, an inline status row below the `<pre>` ("Generating…" with spinner before the first token / "Cancelled" with close icon, `text-warning`), and a **framed footer `Card`** that's the only bordered element — shown on terminal state (body present, stream not live), it stacks the extracted-Jinja-variables chip strip (hidden when none) above the `ActionBar` (Save as template primary / Copy secondary, flips to "Copied" for 1.5s / Clear danger with `SvgDelete`, replacing the old `SvgClose` which read as "close panel"). No "Done" indicator — body being complete + the footer card appearing is enough. Error/idle have their own empty-state UI (use `min-h-[50vh] lg:h-full` so they center without relying on a definite parent height on mobile). Responsive scroll model: on mobile the OUTER container is the scroll area (`overflow-y-auto`, block flow — body + footer scroll together as one content area, no inner "box" scroll now that the panel has no frame); on desktop (`lg:`) it's a flex column where the body is the inner scroll container (`lg:overflow-auto lg:flex-1`) and the footer is pinned below. `use:autoscroll` is on the body and resolves its target (self if scrollable, else nearest scrollable ancestor) so it follows the stream in both layouts. The footer carries `mx-4 lg:mx-0` (aligns with the body's `p-4` on mobile, full-width pinned on desktop) and the outer carries `pb-24 lg:pb-0` so the footer (and the streaming body's tail) sits above the side-panel FAB on mobile; no padding on desktop (FAB is `lg:hidden`). Cancel lives in `PromptSidePanel`'s footer (and the side panel's FAB on mobile while the panel is closed) — NOT here. Mounts `ReasoningCard` at the top of the body so the reasoning lives inside the preview, above where the template text starts streaming.
    ReasoningCard.svelte          # Collapsible card that appears inside `GenerationPreview` when `generation.reasoning` is non-empty. Collapsed: chevron + 'Reasoning:' + last non-empty line truncated with ellipsis. Expanded: scrollable `<pre>` with `use:autoscroll` (from `lib/actions/`) — jumps to bottom on open, then follows streaming. Any scroll-up gesture (mouse, touch, keyboard) pauses auto-scroll; scrolling back to the bottom resumes it.
    PromptHistoryPanel.svelte     # List view of saved prompt-history entries (newest first, server-side cap of 20). Each entry: intent preview (`line-clamp-2` of the server-truncated `intent_preview`), badges for mode (Generate/Refine, accent/purple), focus (system/user/both), example count (`"N examples"`, only when > 0), and `with template` (only when `had_template`). The card itself is display-only — restore + delete are explicit icon buttons in the footer (both always visible; mobile has no hover so neither can be hover-only). Restore uses `SvgHistory` (reused from the section header — semantic match for "restore from history"); shows `SvgSpinner` in place of the icon while the entry is restoring. Delete goes through `confirmDialog.danger`. Refresh button in the header re-fetches the list. A muted note above the list ("Restoring replaces your current form values.") flags that restore overwrites the current form, including the custom template draft. Exposes a `refresh()` method on the instance so the panel can call it after a save without the panel needing to subscribe to a save-completion event.
    index.ts                      # Barrel re-export (`$lib/components/prompts`)
```

## export/

```
  export/                         # Export domain
    ExportDialog.svelte           # Export dialog (backend + draft/caption source selection). Zip exports that bundle images show an image-size estimate (`DatasetInfo.size_bytes`) and confirm via `confirmDialog.warning(...)` above 500 MB.
```

## settings/

```
  settings/                       # Settings domain (dialog + tabs + shared widget + legacy)
    SettingsDialog.svelte         # App settings dialog (tab container)
    GeneralSettings.svelte        # General settings tab (browser notifications, thumbnails)
    SecuritySettings.svelte       # Security settings tab (key-mode switch, password change, YADC_PASSWORD warning)
    CaptionOptionsFields.svelte   # Shared widget — caption option fields (max tokens, image quality, rounds, reasoning, etc.) with optional diff-dot override indicators
    TemplateManager.svelte        # (LEGACY) Full template CRUD panel — superseded by dedicated /templates route
```

## dialogs/

```
  dialogs/                        # Global utility dialogs
    PasswordPromptDialog.svelte   # Global password prompt modal (driven by passwordPrompt.ts store)
```

## icons/

The icons live at `yadc/webui/src/lib/icons/` (sibling of `lib/components/`, not under it).

```
  icons/             # SVG icon components (Svg* prefix). Added manually from Material Symbols; standard viewBox `0 -960 960 960`. Sizing/color come from Tailwind classes via the `class` prop. Includes `SvgChevronDown`/`SvgChevronUp` pair for collapse indicators.
```
