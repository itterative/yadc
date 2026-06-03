---
name: frontend-architecture
description: WebUI frontend directory structure, stores, components, routes, and patterns. Read architecture-overview first if touching backend too.
category: architecture
priority: 2
keep_updated: true
---

# Frontend Architecture

**Stack**: SvelteKit (Svelte 5 + runes) + Tailwind CSS v4 + TypeScript + Zod

**Backend reference**: See `architecture-overview` memory for the Quart API backend (`yadc/api/`).

## Directory Structure

```
yadc/webui/
  src/
    lib/
      index.ts             # Re-exports: storable, async helpers, TypedEventSource, API_BASE, random
      api.ts               # API_BASE constant, structured error response parsing (isAPIErrorResponse / isAPIErrorDetail / apiErrorMessage), and user-friendly HTTP status messages. Exports `ResponseLike` interface so `apiErrorMessage()` works with both native `fetch` `Response` and `UploadResponse`.
      events.ts            # TypedEventSource — SSE with Zod validation
      async.ts             # debounce (async debounce+dedupe), sleep, synchronized, delayed helpers
      notifications.ts     # Browser Notification API helpers (permission, sending, first-use prompt)
      storable.js          # localStorage-backed writable store
      random.ts            # Seeded PRNG for deterministic stub layouts
      upload.ts            # Promise-based XMLHttpRequest wrapper with upload progress tracking and AbortSignal support. Returns `UploadResponse` which implements `ResponseLike`.
      format.ts            # Shared formatting utilities — `formatBytes(bytes)` for human-readable file sizes
      styles/              # Tailwind @layer components
        badges.css         # .badge
        buttons.css        # .btn variants
        forms.css          # .input
        overlays.css       # .dialog-panel, .alert-{error,warning,success,info}
        utilities.css      # .btn-bar
        animations.css     # spin keyframes etc.
      stores/
        dataset/                       # Dataset domain — types + API split
          index.ts                     # Re-exports for `$lib/stores/dataset`
          types.ts                     # DatasetInfo, ImageInfo, ImagePage, CaptionData, HistoryEntry, DatasetUploadResult, UploadConflict, UploadProgressEvent, CaptioningJobInfo, DatasetFolder, DraftSummary, PromptPreview
          api.ts                       # All fetch/CRUD/upload/captioning helpers (debounced via `debounce()`)
        caption/                       # Caption domain — actions + type + persisted settings
          index.ts                     # Re-exports for `$lib/stores/caption`
          actions.ts                   # captionOptions + lastStartedJobId + startBatchCaptioning/captionSingleImage/stopCaptioning (with toasts + optional onError callback)
          options.ts                   # CaptionOptions type (mirrors backend CaptionJobOptions)
          settings.ts                  # Last-used caption settings persisted to localStorage (env, maxTokens, imageQuality, etc.)
        config/                        # Config domain — types + API split
          index.ts                     # Re-exports for `$lib/stores/config`
          types.ts                     # Config + ConfigApi/ConfigPrompt/ConfigSettings/ConfigReasoning/ConfigDatasetEntry + DatasetConfig/Detail + ExportBackend/Result + ConfigHistoryEntry
          api.ts                       # Config CRUD + export API + drafts API. `fetchConfig`/`fetchConfigHistory` debounced.
        env/                           # Env domain — store + API split
          index.ts                     # Re-exports for `$lib/stores/env`
          store.ts                     # `envs` writable + `refreshEnvs` action + types
          api.ts                       # Env CRUD + model fetching + key-mode. `fetchEnvs`/`fetchModels` debounced.
        templates/                     # Templates domain — store + API + pure helper
          index.ts                     # Re-exports for `$lib/stores/templates`
          store.ts                     # `templates` writable + `refreshTemplates` action + types
          api.ts                       # Template CRUD. `fetchTemplates`/`fetchTemplate` debounced.
          jinja.ts                     # Pure `extractVariables()` (Jinja2 regex helpers)
        events.ts                      # Self-connecting SSE store — opens TypedEventSource on load, pipes events into readonly writable stores (captioningStatus, pendingDatasetChanges, storedCaptions). Auto-refreshes envs/templates on change events. Caption text cached from SSE events (LRU).
        toasts.ts                      # Toast notification store + `toast.{info,success,warning,error}()` helpers
        confirm.ts                     # Promise-based confirmation dialog store
        passwordPrompt.ts              # Global password prompt store + `withPasswordRetry` + `PasswordPromptCancelled`
        sessionPassword.ts             # Tab-scoped in-memory password store
        storageStore.ts                # Generic `localStorage`/`sessionStorage`-backed writable factory
        settings.ts                    # UI settings (storable) + `settingsDialog` open-state
        topbar.svelte.ts               # Topbar `Snippet` shared via $state
      components/
        ui/                            # Atomic, reusable primitives (no domain logic)
          Dialog.svelte                # Modal dialog (HTML <dialog>)
          Alert.svelte                 # Inline alert (info/warning/error/success, dismissable, optional actions snippet)
          FileDropZone.svelte          # Drag-and-drop + file/folder picker with client-side validation, recursive directory traversal, and `allowedExtensions` filtering
          Checkbox.svelte              # Checkbox component
          CodeMirror.svelte            # CodeMirror 6 wrapper (Svelte 5 runes, doc/ext sync)
          JinjaEditor.svelte           # Jinja2 template editor (CM6 + @codemirror/lang-jinja). Bindable `value` + `variables` (extracted template vars)
          TomlEditor.svelte            # TOML editor (CM6 + @codemirror/legacy-modes, optional readonly mode). Bindable `value`
          IntersectionObserverElement.svelte  # Infinite scroll sentinel
          tabs/                        # Tab system (Svelte 5 context + snippets)
            Tabs.svelte                # Generic container — tab bar layout, registration, bindable value
            PillTabs.svelte            # Pre-styled pill variant (wraps Tabs with rounded-full buttons)
            CompactPillTabs.svelte     # Compact segmented control variant (bg-gray-800/50 container, rounded-md buttons, icon support)
            Tab.svelte                 # Child that auto-registers via context, shows/hides content. Supports optional icon prop.
            TabsContext.svelte.ts      # Symbol key + state factory + typed helpers. TabItem has optional icon (Component).
          ConfirmDialog.svelte         # Global confirmation dialog (Promise-based, mounted in layout, supports string + snippet body, variant: danger/warning/info)
          SpinnerBlock.svelte          # Centered spinner with optional label and size
          PromptPreview.svelte         # Self-contained prompt preview (template selector + system/user prompt display)
          Tooltip.svelte               # Pure-CSS hover tooltip (wraps a trigger, shows label to the right on hover)
          SetTopbar.svelte             # Sets the layout topbar snippet from a page component (lifecycle-managed via $effect)
          ToastContainer.svelte        # Fixed-position toast stack (mounted in +layout.svelte)
          ToastItem.svelte             # Single toast (message, variant, progress bar, dismiss, optional action button)
          Card.svelte                  # Generic card wrapper (`rounded-lg bg-gray-800`). Body and footer (typically `ActionBar`) go in `children`.
          ActionBar.svelte             # Footer flex container (`flex gap-2 bg-black/15`) for card action buttons. Items distribute evenly via `flex-1`.
          ActionBarItem.svelte         # Standardized action button inside ActionBar — accepts icon, variant (`primary`/`secondary`/`danger`).
        dataset/                        # Dataset domain — one sub-folder per feature
          browser/                      # Image grid + tile
            DatasetBrowser.svelte       # Masonry grid container (column distribution + infinite scroll + selectedId)
            DatasetImage.svelte         # Masonry grid tile (thumbnail + badges + selected outline)
          detail/                       # Image detail (tab host + tabs)
            ImageDetail.svelte          # Tab system host (CompactPillTabs: Caption / Preview / Edit) + data layer (captionData / historyEntries / API calls). History auto-loaded alongside caption.
            Caption.svelte              # Caption box (ActionCard + ActionBar) + Drafts (Card+ActionBar: Promote/Delete) + History (always visible when entries exist, Restore/Delete by content hash)
            Preview.svelte              # Thin wrapper around PromptPreview
            Extras.svelte               # Always-editable TOML editor for the image's extras_raw (ActionCard + ActionBar with Save/Cancel)
          config/                       # Dataset config editor (tab host + tabs + helpers)
            DatasetConfig.svelte        # Tab system host (Form / Advanced / History) + data fetching + save orchestration
            DatasetConfigForm.svelte    # Structured form view of the config — fields bound to `configState` Svelte 5 `$state` rune from `state.svelte.ts`
            DatasetConfigAdvanced.svelte # Raw TOML editor view of the config. Binds to `configState.rawContent`; shows a warning when the structured form has unsaved changes.
            ConfigHistory.svelte        # Config revision history browser (list, view, restore entries). Uses ActionCard + ActionBar per entry. Reads from `ConfigHistoryRepository` on the backend.
            patch.ts                    # Pure TOML patch helpers (e.g. `assemblePatchFromForm`) — no Svelte/DOM dependencies, testable in isolation.
            state.svelte.ts             # Svelte 5 `$state` rune module holding the form / raw content state shared between `DatasetConfigForm` and `DatasetConfigAdvanced`. Exposes `simplifiedDirty` flag.
          upload/                       # Dataset upload (panel + slim tab wrappers)
            DatasetUploadPanel.svelte   # Shared upload panel (create + append modes). When `mode="append"`, accepts an `initialFiles` prop (consumed once via `untrack()`) to pre-populate the file list from a drop event on the page.
            UploadDatasetTab.svelte     # Upload tab — file selection, progress bar, cancel upload, toast warnings
            CreateDatasetTab.svelte     # Create/Import tab — radio toggle between "Add image paths" and "Import TOML config"
          manage/                       # Dataset manage (delete + folder ops)
            DatasetManageTab.svelte     # "Manage" tab inside the dataset creation dialog — list managed (upload-sourced) datasets with folder/image counts and delete actions.
        caption/                        # Caption domain
          CaptionSettingsPanel.svelte   # The batch-captioning side panel. Derives isBatchCaptioning from store, shows Start/Stop button. Writes assembled options to captionActions store reactively. Fetches dataset config defaults, shows diff dots for overrides. Auto-saves overrides to `captionSettings` localStorage via `deferred()` on any field change; flushes pending save before starting captioning.
        env/                            # Env domain
          EnvironmentSettings.svelte    # Environment CRUD tab (env list, inline edit, token reveal)
          EnvSelector.svelte            # Environment form (env dropdown + URL/token/model, bindable props). "Manage…" link opens SettingsDialog at the Environments tab.
        export/                         # Export domain
          ExportDialog.svelte           # Export dialog (backend + draft/caption source selection)
        settings/                       # Settings domain (dialog + tabs + shared widget + legacy)
          SettingsDialog.svelte         # App settings dialog (tab container)
          GeneralSettings.svelte        # General settings tab (browser notifications, thumbnails)
          SecuritySettings.svelte       # Security settings tab (key-mode switch, password change, YADC_PASSWORD warning)
          CaptionOptionsFields.svelte   # Shared widget — caption option fields (max tokens, image quality, rounds, reasoning, etc.) with optional diff-dot override indicators
          TemplateManager.svelte        # (LEGACY) Full template CRUD panel — superseded by dedicated /templates route
        dialogs/                        # Global utility dialogs
          PasswordPromptDialog.svelte   # Global password prompt modal (driven by passwordPrompt.ts store)
        icons/             # SVG icon components (Svg* prefix). Added manually from Material Symbols; standard viewBox `0 -960 960 960`. Sizing/color come from Tailwind classes via the `class` prop.
    routes/
      layout.css        # Tailwind v4 imports + @theme block + typography plugin
      +layout.svelte    # App shell with left sidebar nav (icon-rail on desktop, slide-in overlay on mobile with burger menu). Brand uses android-chrome-192x192.png icon. Tooltip component for desktop hover labels.
      +page.svelte      # Dataset listing (cards with edit/delete, add-dataset dashed card) → links to #/datasets/{name}
      AddDatasetDialog.svelte   # Co-located: dataset creation dialog shell. Two tabs: Upload (via UploadDatasetTab) and Create/Import (via CreateDatasetTab)
      EditDatasetDialog.svelte  # Co-located: edit dataset dialog (Config / Manage / Upload tabs)
      templates/
        +page.svelte              # Template listing (grid cards with edit/delete, add-template dashed card) — mirrors dataset listing
        EditTemplateDialog.svelte # Co-located: create/edit template dialog (JinjaEditor)
      datasets/[name]/
        +page.svelte              # Dataset browser (masonry grid + side panel + captioning progress in stats line + topbar upload icon)
        SidePanel.svelte          # Co-located: tabbed side panel (caption/details/config) with mobile drawer. Minimal prop threading — captioning actions handled by components via stores.
        AddFilesDialog.svelte     # Co-located: dialog wrapping DatasetUploadPanel in `mode="append"` for adding files to an existing managed dataset. Pre-populates the file list from dropped files.
        DropUploadZone.svelte     # Co-located: drop-target wrapper with a slot. Owns drag/drop state + `webkitGetAsEntry` file collection. Renders a full-area overlay (accent for allowed drops, warning for blocked). Emits `ondrop(files)` / `onblockeddrop(reason)`. Reused only here today — co-locate until a second consumer appears.
```

## Component Organization

**Where a component lives** is decided by two rules:

1. **Scope rule (placement):**
   - `lib/components/ui/` — atomic UI primitives, no domain concepts, no API calls.
   - `lib/components/<domain>/` — domain components, used in ≥2 places OR used by a `lib/` component.
   - `routes/<path>/` (co-located) — used by exactly one route AND not by anything in `lib/`. **Promote to `lib/` on the second consumer.**
2. **Feature-folder rule (sub-folder for parent + children):** A feature gets its own sub-folder when it has ≥3 related `.svelte` files forming a coherent unit (a host component + its tabs/sections), all of which are used only by that host. Folder name = feature name. The host file shares the folder name. Inside the folder: `Feature.svelte`, `FeatureTab.svelte`, `state.svelte.ts` (if shared runes), `helpers.ts` (if pure helpers used by multiple files in the folder). Pure helpers used **outside** the folder stay in `lib/` proper.

**Current `lib/components/` layout:**

- `ui/` — atomic primitives: `Dialog`, `Alert`, `Tabs`/`Tab`/`PillTabs`/`CompactPillTabs` (in `tabs/` sub-folder), `CodeMirror`, `SpinnerBlock`, `ToastContainer`/`ToastItem`, `Tooltip`, `FileDropZone`, `Checkbox`, `KeyValueEditor`, `PromptPreview`, `Topbar`, `Card`, `ActionBar`/`ActionBarItem`, `IntersectionObserverElement`.
- `dataset/` — dataset domain. Sub-folders per feature:
  - `browser/` — `DatasetBrowser`, `DatasetImage` (masonry grid + tile)
  - `detail/` — `ImageDetail` (tab host) + `Caption`, `Preview`, `Extras` (tabs)
  - `config/` — `DatasetConfig` (tab host) + `DatasetConfigForm`/`DatasetConfigAdvanced`/`ConfigHistory` (tabs) + `patch.ts` + `state.svelte.ts`
  - `upload/` — `DatasetUploadPanel` (the main create+append+conflict panel) + `UploadDatasetTab`/`CreateDatasetTab` (slim wrappers for dialog entry points)
  - `manage/` — `DatasetManageTab` (manage tab in the Edit dialog)
- `caption/` — caption domain: `CaptionSettingsPanel` (the batch-captioning side panel — promoted from `routes/datasets/[name]/`).
- `env/` — env domain: `EnvironmentSettings` (CRUD list + edit), `EnvSelector` (env picker for the caption flow).
- `export/` — export domain: `ExportDialog` (export dialog).
- `settings/` — settings domain: `SettingsDialog` (host) + `GeneralSettings`/`EnvironmentSettings`-via-env/`SecuritySettings` (tabs) + `CaptionOptionsFields` (settings widget). `TemplateManager.svelte` is also here, **marked LEGACY** — superseded by the dedicated `/templates` route + `EditTemplateDialog`.
- `dialogs/` — only `PasswordPromptDialog` (global utility dialog mounted in `+layout.svelte`).
- `icons/` — `Svg*` SVG components, all with `viewBox="0 -960 960 960"`, sized/colored via Tailwind via the `class` prop.

**Imports:** `$lib/` absolute paths for cross-boundary refs, `./` relative for co-located files. No file in `lib/` imports from `routes/` — preserved by the "promote on second use" rule.

**Route-only components** (co-located next to the route that uses them, not in `src/lib/`):

- `routes/+layout.svelte` — app shell with left sidebar + topbar + global dialogs.
- `routes/+page.svelte` — dataset listing (cards with edit/delete, add-dataset dashed card).
- `routes/AddDatasetDialog.svelte` — co-located: dataset creation dialog shell (Upload / Create tabs).
- `routes/EditDatasetDialog.svelte` — co-located: edit dataset dialog (Config / Manage / Upload tabs).
- `routes/datasets/[name]/+page.svelte` — dataset browser page.
- `routes/datasets/[name]/SidePanel.svelte` — co-located: tabbed side panel (caption/details/config).
- `routes/datasets/[name]/AddFilesDialog.svelte` — co-located: dialog wrapping `dataset/upload/DatasetUploadPanel` in `mode="append"`.
- `routes/datasets/[name]/DropUploadZone.svelte` — co-located: drop-target wrapper with a slot. **Owns drag/drop state + `webkitGetAsEntry` file collection. Reused only here today — co-locate until a second consumer appears.**

## Store Organization

Mirrors the component organization conventions. Two rules:

1. **Scope rule (placement):**
   - `lib/stores/<domain>/` — stores serving a single domain (dataset, caption, config, env, templates). Each sub-folder has a re-exporting `index.ts` so callers import from `$lib/stores/<domain>`.
   - `lib/stores/` (top-level) — global UI primitives (toasts, confirm, password dialog, session password, settings, topbar) and the SSE event backbone. Singletons that don't form a coherent feature.
2. **Sub-folder rule (when to introduce a domain sub-folder):** A domain gets its own sub-folder when it has ≥2 related files (types + API + store + actions) AND they all serve the same domain. Inside the folder: `index.ts` (re-exporting barrel for back-compat), `types.ts` (data shapes), `api.ts` (transport — fetch/CRUD), plus `store.ts`/`actions.ts` when reactive state is involved, plus pure helpers (e.g. `jinja.ts`).

**Current `lib/stores/` layout:**

- `dataset/` — types (`types.ts`) + API (`api.ts`) + re-export (`index.ts`).
- `caption/` — actions (`actions.ts`) + options type (`options.ts`) + persisted settings (`settings.ts`) + re-export.
- `config/` — types + API + re-export.
- `env/` — store (`store.ts`) + API (`api.ts`) + re-export.
- `templates/` — store + API + pure `jinja.ts` helper + re-export.
- Top-level (singletons, no sub-folder): `events.ts`, `toasts.ts`, `confirm.ts`, `passwordPrompt.ts`, `sessionPassword.ts`, `settings.ts`, `topbar.svelte.ts`, `storageStore.ts`.

**Imports:** `$lib/stores/<domain>` (re-exporting barrel) for cross-boundary refs, `./<file>` relative for files inside the same sub-folder. No file in `lib/` imports from `routes/` — preserved by the existing rule.

## Store Modules

The 16 flat store files are organized into **5 domain sub-folders** (each with a re-exporting `index.ts` so callers import from `$lib/stores/<domain>`) and **7 top-level files** (global UI primitives + the SSE event store). The old `captioning.ts` re-export shim was dropped (0 callers).

| File | Purpose |
|------|---------|
| `dataset/` | Dataset domain. `types.ts` (DatasetInfo, ImageInfo, ImagePage, CaptionData, HistoryEntry, DatasetUploadResult, UploadConflict, UploadProgressEvent, CaptioningJobInfo, DatasetFolder, DraftSummary, PromptPreview) + `api.ts` (all fetch/CRUD/upload/captioning helpers, debounced via `debounce()`). Includes `deleteDataset`, `uploadDataset()` (create), `appendUploadDataset()` (append to managed), `commitStagingUpload()` (resolve conflicts), `startCaptioning`/`stopCaptioning`/`captionSingleImage` (raw API). |
| `caption/` | Caption domain. `actions.ts` (captionOptions writable, startBatchCaptioning/captionSingleImage/stopCaptioning with toasts + optional onError callback, lastStartedJobId) + `options.ts` (CaptionOptions type mirroring backend CaptionJobOptions) + `settings.ts` (last-used caption settings persisted to localStorage — env, maxTokens, imageQuality, etc. — restored on panel open, saved automatically on field change via `deferred()` (300 ms); priority: localStorage override → dataset config default → hardcoded default). |
| `config/` | Config domain. `types.ts` (Config + nested ConfigApi/ConfigPrompt/ConfigSettings/ConfigReasoning/ConfigDatasetEntry + DatasetConfig/Detail + ExportBackend/Result + ConfigHistoryEntry) + `api.ts` (config CRUD + export API + drafts API; `fetchConfig`/`fetchConfigHistory` debounced). |
| `env/` | Env domain. `store.ts` (`envs` writable holding full `EnvInfo[]` details from `GET /api/envs`, not just names; `refreshEnvs` action) + `api.ts` (env CRUD + model fetching + key-mode; `fetchEnvs`/`fetchModels` debounced). |
| `templates/` | Templates domain. `store.ts` (`templates` writable; `refreshTemplates` action) + `api.ts` (template CRUD; `fetchTemplates`/`fetchTemplate` debounced) + `jinja.ts` (pure `extractVariables()` — Jinja2 regex helpers, no API/store dependencies). |
| `events.ts` | Self-connecting SSE store — opens `TypedEventSource` on module load (browser), validates with Zod, pipes into `readonly` writable stores. Exports `captioningStatus`, `pendingDatasetChanges`, `resumptionFailed`, `lastCaptionedImage`, `lastCaptionError`, `currentlyCaptioning`, `storedCaptions` (LRU cache of captions from SSE), `getStoredCaption()`, `clearStoredCaption()`, `clearPendingDatasetChange()`, `clearResumptionFailed()`, `setCaptioningStatus()`, `setCurrentlyCaptioning()`. Listens for `environments_changed` and `templates_changed` events to auto-refresh their stores. Uses browser's built-in `EventSource` auto-reconnect (preserves `Last-Event-ID`). |
| `toasts.ts` | Toast notification store — manages a reactive list of active toasts with auto-dismiss. Exports `toasts` readable store, `addToast()`, `dismissToast()`, and `toast.success/error/warning/info()` convenience helpers. |
| `confirm.ts` | Promise-based confirmation dialog store — manages dialog state with `confirm()` → `Promise<boolean>`. Exports `confirmState` readable store, `confirmDialog.danger/warning/info()` convenience helpers. Supports string messages and Svelte snippet bodies. |
| `passwordPrompt.ts` | Global password prompt store — `requestPassword()` returns `Promise<string>`, `withPasswordRetry(action)` catches `PasswordRequiredError` and retries once after dialog. Module-level `pendingPromise` deduplicates concurrent callers. |
| `sessionPassword.ts` | Tab-scoped in-memory password store (Svelte writable, never persisted to localStorage). Captioning requests read from it automatically. |
| `settings.ts` | UI settings (localStorage) — `notifications` tri-state (`"unset"` / `"enabled"` / `"disabled"`). `settingsDialog` store tracks open state + active tab (`general`/`environments`/`security`). |
| `topbar.svelte.ts` | `$state` module holding the current page's topbar `Snippet`. Pages set it via `<SetTopbar>` (which uses `$effect` lifecycle to set/clear). Layout reads it with `getTopbarContent()` and renders with `{@render}`. |
| `storageStore.ts` | Generic `localStorage`/`sessionStorage`-backed writable store factory. |

## Key Patterns

- **Hash routing**: SvelteKit uses `router: { type: "hash" }` — all internal links use `#/` prefix. Quart only serves `GET /` + static assets.
- **SSE**: `stores/events.ts` is a self-connecting store module. Opens `TypedEventSource` on module load, validates events with Zod, pipes into `readonly` writable stores. The browser's built-in `EventSource` auto-reconnect handles reconnection automatically — it preserves and sends `Last-Event-ID` on reconnect, allowing the backend to replay missed events from its ring buffer. For manual reconnections (e.g. mobile visibility recovery, fallback interval), `TypedEventSource.lastEventId` is tracked and passed as a `?lastEventId=` query parameter. A `visibilitychange` listener detects mobile background/foreground transitions: if hidden for > 5 s, the connection is force-closed and reconnected with the tracked event ID. The `onerror` handler captures `lastEventId` before the EventSource becomes unusable, but does **not** call `close()` (which would discard the internal last-event-id state). A 5s interval fallback reconnect handles edge cases where the EventSource ends up in CLOSED state. If resumption fails (history too old), a `resumption_failed` event sets a `resumptionFailed` store, and a warning toast is fired (plus inline banner on the dataset detail page).
- **Dataset watcher**: Backend emits `DatasetChangedEvent` via SSE when filesystem changes are detected. Frontend stores these in `pendingDatasetChanges` (a `Set<string>`). Dataset browser page subscribes and shows a "Refresh" banner.
- **Drop-to-upload in the dataset browser**: A full-area `DropUploadZone.svelte` wrapper owns the drag/drop state and renders a full-area overlay (accent for allowed drops, warning for blocked). On a valid drop, the page receives the files via `ondrop(files)` callback and opens `AddFilesDialog` pre-populated with the dropped files. The dialog wraps `DatasetUploadPanel` in `mode="append"`, and the panel's new `initialFiles` prop (consumed once via `untrack()`) seeds the file list. An upload icon in the topbar opens the same dialog with no pre-populated files. Uploads are only allowed when `currentDataset.source === "upload"` and no batch captioning is running — the overlay shows a warning in that case (`uploadBlockedReason` is surfaced via `onblockeddrop(reason)` → toast).
- **CodeMirror 6**: `CodeMirror.svelte` wrapper uses three separate `$effect` blocks (create/destroy/sync) — never combine. Uses `editable` prop (default `true`) — not `readonly`. Includes a `baseTheme` (dark surface, accent-colored selection via `color-mix(in oklch, ...)`) and a `darkHighlightStyle` that maps all `@lezer/highlight` tags to CSS `--color-syn-*` variables defined in the Tailwind `@theme` block.
- **Svelte 5**: No pipe directives on events. No nested `<button>`. Use `<div role="button">` for clickable list items.
- **Tabs**: `ui/tabs/` uses Svelte context (`TabsContext.svelte.ts` with Symbol key) for parent-child coordination. `Tab` children register themselves during init via `untrack()` (synchronous, before parent renders). `Tabs` supports a custom `tab` snippet for arbitrary button styling; `PillTabs` is a pre-styled variant (rounded-full buttons on border-bottom bar). `CompactPillTabs` is a compact segmented-control variant (bg-gray-800/50 container, rounded-md buttons) with icon support — uses `TabsContext` directly rather than wrapping `Tabs`. `Tab` accepts optional `icon` prop (Svelte `Component<{ class?: string }>`). `value` is bindable and uses `id` (not `label`) for matching.
- **Tailwind v4**: Custom colors must be registered in `@theme { }` block, not `:root` vars.
- **Tailwind content detection**: Root `.gitignore` `lib/` rule was hiding `src/lib/` — fixed with `!src/lib/` in `webui/.gitignore`.
- **npm security**: `min-release-age=14` in `.npmrc` blocks installing packages published <14 days ago.
- **Z-index layers**: Fixed-position elements use a consistent z-index stack:
  - `z-10`: Local absolute-positioned overlays within components (e.g. ImageDetail hover/delete masks)
  - `z-20`: FABs (mobile caption settings floating button)
  - `z-30`: Mobile overlay backdrops (side panel scrim, sidebar scrim)
  - `z-40`: Mobile slide-in panels (side panel drawer, sidebar drawer on small screens)
  - `z-50`: Global overlays — dialogs (`Dialog.svelte`)
  - `z-60`: Toast notifications (`ToastContainer.svelte`) — above dialogs so they remain visible
  - When adding new fixed/absolute layers, use the appropriate slot and avoid values outside this scale.
- **Topbar pattern**: The layout has a topbar (inside `app-content`, between sidebar and main). Pages set topbar content by defining a `{#snippet}` and passing it to `<SetTopbar>`. The snippet is stored in `topbar.svelte.ts` (a `$state` module). The layout reads it with `getTopbarContent()` and renders with `{@render}`. On mobile the topbar also houses the burger menu button. On desktop, if no snippet is set, the topbar is hidden via `:empty`.
- **Browser notifications**: `notifications.ts` is the single gatekeeper. `sendNotification()` checks support, settings preference (`notifications === "enabled"`), browser permission, and tab visibility — callers just call it with no pre-checks. `promptNotificationsOnce()` shows a one-time toast with "Enable" action on first captioning start (session-guarded, only when `notifications === "unset"`). Settings dialog General tab has a checkbox that toggles between `"enabled"`/`"disabled"`. Global notification dispatching lives in `+layout.svelte` so it works even when the user navigates away from the dataset page.

## Styling Patterns

Three-tier approach for mixing Tailwind utilities, reusable component classes, and scoped Svelte styles:

### 1. `@layer components` in CSS files (`src/lib/styles/`)

For **reusable UI abstractions** that appear across multiple components. Use `@apply` to compose Tailwind utilities into named classes.

- `buttons.css` — `.btn`, `.btn-primary`, `.btn-secondary`
- `forms.css` — `.input`, `.label`
- `badges.css` — `.badge`, `.badge-sm`, `.badge-accent`
- `utilities.css` — `.diff-dot`, `.card`, `.section-heading`

**Rules**:
- Extract appearance only (colors, typography, border-radius, padding). **Do not include positioning** (absolute, fixed, margins for layout) — that stays inline in the consuming component.
- Keep semantic names that describe what the thing *is*, not where it sits.

### 2. Inline Tailwind utilities in HTML

For **one-off structural and layout styles** that are specific to a single element and not worth naming.

Examples:
- `class="h-full overflow-hidden"` on a CodeMirror wrapper
- `class="min-w-0 overflow-hidden"` on a flex child that needs to shrink
- `class="max-sm:gap-3 max-sm:px-4"` for responsive tweaks
- `class="absolute top-1 right-1"` for positioning a badge inside a card

### 3. Svelte `<style>` scoped blocks

For styles that genuinely can't be utilities:

- **State classes** applied via `class:selected` or similar — e.g. `.selected { outline: 2px solid var(--color-accent); }`
- **JS-driven CSS variables** — e.g. `.grid-cols-auto { grid-template-columns: repeat(var(--x-grid-cols), minmax(0, 1fr)); }` where `--x-grid-cols` is set via inline style from JS
- **Complex structural layout** that's easier to read as CSS — e.g. the navbar layout in `+layout.svelte`

**Anti-patterns to avoid**:
- Using `@apply` inside Svelte `<style>` blocks — it adds indirection without benefit. Either extract to `src/lib/styles/` (if reusable) or inline utilities (if one-off).
- Creating component classes that include positioning (e.g. `.badge-corner` with `absolute`). Positioning is context-specific and should be inline.
- Writing raw CSS for styles that map directly to existing utilities (e.g. `display: flex; align-items: center;` instead of `class="flex items-center"`).
