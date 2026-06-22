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
    EditTemplateDialog.svelte     # Reusable create/edit template dialog. `templateName=null` for new; otherwise edits the named template. Used by both the `/templates` route and `caption/TemplateSection.svelte`. `initialContent?` pre-fills the editor for new templates — used by the prompt generator to save a generated body.
    index.ts                      # Barrel re-export (`$lib/components/templates`)
```

## prompts/

```
  prompts/                        # Prompt-generator domain — feature folder
    PromptGenerator.svelte        # Host for the prompts route. Layout: full-page `GenerationPreview` on the left + `PromptSidePanel` (form/settings/history tabs + Generate / Save-to-history footer) on the right. Owns the form-state `$state`s (mode, env, apiUrl, apiToken, apiModelName, intent, focus, maxTokens, imageQuality, templateContent, examples) and the persistence `$effect` that mirrors them into `promptSettings` on every change. Also owns the `EditTemplateDialog` flow — the preview's "Save as template" button opens this dialog (it's not triggered from the side panel). `ongenerate` callback (passed to the panel) calls `startGeneration` with a snapshot of form values so mid-stream edits don't leak into the in-flight call. Re-derives `canGenerate` locally as a defense-in-depth guard for `handleGenerate` (the panel derives its own copy to drive its button).
    PromptSidePanel.svelte        # Tabbed side panel hosting the three settings tabs (Generate / Settings / History) and the footer actions: Save-to-history + Generate/Refine (or Cancel while streaming). Built on the generic `ui/SidePanel` primitive — desktop inline (`lg:static lg:w-120`), mobile slide-over drawer whose `fab` snippet swaps the FAB between a streaming cancel and the default toggle (via `FabButton`). The primary CTA in the footer swaps to Cancel mid-stream so the user can stop the in-flight call from inside the panel (the FAB provides the same on mobile while the panel is closed). Form state is bound through from the host (`bind:mode`, `bind:intent`, etc.); `open` and `activeTab` are also bindable so the mobile X (in the `PillTabs` `end` snippet) can close the drawer and so the panel's restore handler can switch back to the Generate tab. Owns the `historyPanel` ref + `isSaving` state + `handleSaveToHistory` + `handleRestore`. Reads `generation.status` directly from the store to derive `isStreaming` / `canGenerate` / `canSave` for the footer buttons.
    PromptForm.svelte             # The Generate-tab form: intent textarea, focus toggle, mode toggle (New/Refine pills, bound to host `mode` via `bind:mode`), template section (refine only), few-shot examples section. Selecting Refine reveals the Template section (picker `<select>` over `$templates.items` + `JinjaEditor` + variables chip strip) between the mode pills and the examples, and relabels the intent field ("Refinement intent"). Env/model/generation-limits live in `PromptSettings` (the Settings tab), NOT here. Rendered once by the host, so internal state (template picker selection, loading flags) survives a mode switch.
    PromptSettings.svelte         # The Settings-tab form: Environment section (env `<select>` + API URL + API token + model picker with refresh, auto pre-filled from the selected env) moved out of `PromptForm`, plus a Generation-limits section — `max_tokens` (number input, default 16384; the client default of 4096 truncates longer templates) and `image_quality` (auto/low/high pills, few-shot example fidelity → OpenAI `image_url.detail` / Gemini `mediaResolution`). All fields bindable against the host's persisted `promptSettings`; disables its inputs while a generation is streaming. Mirrors `caption/EnvSelector` for the env/model part.
    ExamplesPanel.svelte          # Few-shot examples list with thumbnail + subject/caption fields. Two add modes: `+ Manual` (file picker → FileReader → data URL) and `+ From dataset` (inline picker with dataset selector + image grid; auto-fills subject=file_stem, caption=dataset caption).
    GenerationPreview.svelte      # Streaming monospace preview with blinking caret, copy-to-clipboard, Save-as-template button, extracted Jinja variables as chips (in a footer that's hidden when no variables), and an inline status row below the `<pre>` ("Generating…" with spinner before the first token / "Cancelled" with close icon, `text-warning`). No "Done" indicator — body being complete + Copy/Save buttons appearing in the header is enough. Error has its own full empty-state UI in the body. Cancel lives in `PromptSidePanel`'s footer (and the side panel's FAB on mobile while the panel is closed) — NOT here. Mounts `ReasoningCard` at the top of the body so the reasoning lives inside the preview, above where the template text starts streaming.
    ReasoningCard.svelte          # Collapsible card that appears inside `GenerationPreview` when `generation.reasoning` is non-empty. Collapsed: chevron + 'Reasoning:' + last non-empty line truncated with ellipsis. Expanded: scrollable `<pre>` that auto-follows streaming (only if user is within 50px of the bottom — leaves them alone if they scrolled up to read).
    PromptHistoryPanel.svelte     # List view of saved prompt-history entries (newest first, server-side cap of 20). Each entry: intent preview (`line-clamp-2` of the server-truncated `intent_preview`), badges for mode (Generate/Refine, accent/purple), focus (system/user/both), example count (`"N examples"`, only when > 0), and `with template` (only when `had_template`). The card itself is display-only — restore + delete are explicit icon buttons in the footer (both always visible; mobile has no hover so neither can be hover-only). Restore uses `SvgHistory` (reused from the section header — semantic match for "restore from history"); shows `SvgSpinner` in place of the icon while the entry is restoring. Delete goes through `confirmDialog.danger`. Refresh button in the header re-fetches the list. Exposes a `refresh()` method on the instance so the panel can call it after a save without the panel needing to subscribe to a save-completion event.
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
