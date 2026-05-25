---
name: frontend-architecture
description: Frontend (webui) architecture — directory structure, stores, components, routes, and key patterns.
---

# Frontend Architecture

**Stack**: SvelteKit (Svelte 5 + runes) + Tailwind CSS v4 + TypeScript + Zod

**Backend reference**: See `architecture-overview` memory for the Flask API backend (`yadc/api/`).

## Directory Structure

```
yadc/webui/
  src/
    lib/
      index.ts             # Re-exports: storable, async helpers, TypedEventSource, API_BASE, random
      api.ts               # API_BASE constant (empty in prod, backend URL in dev)
      events.ts            # TypedEventSource — SSE with Zod validation
      async.ts             # deferred, sleep, synchronized, delayed helpers
      notifications.ts   # Browser Notification API helpers (permission, sending, first-use prompt)
      storable.js          # localStorage-backed writable store
      random.ts            # Seeded PRNG for deterministic stub layouts
      styles/              # Tailwind @layer components
        badges.css         # .badge
        buttons.css        # .btn variants
        forms.css          # .input
        overlays.css       # .dialog-panel
        utilities.css      # .btn-bar
      stores/
        settings.ts       # UI settings (storable)
        events.ts         # Self-connecting SSE store — opens TypedEventSource on load, pipes events into readonly writable stores (captioningStatus, pendingDatasetChanges)
        captioning.ts     # Re-export shim from events.ts for backward compatibility
        captionOptions.ts # CaptionJobOptions type for caption settings dialog
        datasetImages.ts  # Types (DatasetInfo, ImageInfo, ImagePage, CaptionData) + API helpers + deleteDataset
        configs.ts        # Config CRUD API helpers + export backend types
        envs.ts           # Environment CRUD API helpers + types (EnvInfo)
        templates.ts      # Template CRUD API helpers + types (TemplateInfo)
      components/
        ui/                           # Atomic, reusable primitives
          Dialog.svelte               # Modal dialog (HTML <dialog>)
          Checkbox.svelte             # Checkbox component
          CodeMirror.svelte           # CodeMirror 6 wrapper (Svelte 5 runes, doc/ext sync)
          JinjaEditor.svelte          # Jinja2 template editor (CM6 + @codemirror/lang-jinja)
          TomlEditor.svelte           # TOML editor (CM6 + @codemirror/legacy-modes, optional readonly mode)
          IntersectionObserverElement.svelte  # Infinite scroll sentinel
          TabBar.svelte               # Configurable tab bar with active state styling
          ConfirmDelete.svelte        # Delete confirmation block (cancel/confirm buttons)
          SpinnerBlock.svelte         # Centered spinner with optional label and size
          PromptPreview.svelte        # Self-contained prompt preview (template selector + system/user prompt display)
          ToastContainer.svelte       # Fixed-position toast stack (mounted in +layout.svelte)
          ToastItem.svelte            # Single toast (message, variant, progress bar, dismiss, optional action button)
        dataset/                        # Dataset-domain components
          DatasetImage.svelte         # Masonry grid tile (thumbnail + badges + selected outline)
          DatasetBrowser.svelte       # Masonry grid container (column distribution + infinite scroll + selectedId)
          ImageDetail.svelte          # Image detail side panel (full image + caption edit + TOML viewer + drafts + PromptPreview)
        dialogs/                        # Dialog-shaped components
          EnvManager.svelte           # Environment CRUD dialog
          ExportDialog.svelte         # Export dialog (backend + draft/caption source selection)
          SettingsDialog.svelte       # App settings dialog (General tab + EnvManager launcher)
        settings/                       # Settings-domain sub-components
          EnvSelector.svelte          # Environment form (env dropdown + URL/token/model, bindable props, reload trigger)
          TemplateManager.svelte      # (LEGACY) Full template CRUD panel — now superseded by dedicated /templates route
      icons/             # SVG icon components (Svg* prefix)
    routes/
      layout.css        # Tailwind v4 imports + @theme block + typography plugin
      +layout.svelte    # App shell with breadcrumb nav (hash routing links) + global captioning notification watcher
      +page.svelte      # Dataset listing (cards with edit/delete, add-dataset dashed card) → links to #/datasets/{name}
      AddDatasetDialog.svelte   # Co-located: create/import dataset dialog (used only by +page.svelte)
      EditDatasetDialog.svelte  # Co-located: edit dataset TOML config dialog (CodeMirror TOML editor)
      templates/
        +page.svelte              # Template listing (grid cards with edit/delete, add-template dashed card) — mirrors dataset listing
        EditTemplateDialog.svelte # Co-located: create/edit template dialog (JinjaEditor)
      datasets/[name]/
        +page.svelte              # Dataset browser (masonry grid + side panel + captioning progress in stats line)
        CaptionSettings.svelte    # Co-located: captioning settings side panel. Fetches dataset config defaults from GET /configs/<name>, pre-fills fields, shows diff dots for overridden values, collapsible Overrides section with per-field reset.
        DatasetConfig.svelte      # Co-located: dataset config editor tab. Structured form for caption settings (max_tokens, image_quality, rounds, overwrite, reasoning, prompt) saved via PATCH /configs/<name>. Read-only raw TOML view at bottom.
        SidePanel.svelte          # Co-located: tabbed side panel (caption/details/config) with mobile drawer
```

## Component Organization

- `ui/` — atomic reusable primitives, no domain logic
- `dataset/` — dataset-domain components shared across routes
- `dialogs/` — dialog-shaped components (opened from layout/nav or multiple places)
- `settings/` — sub-components shared by SettingsDialog and CaptionSettings
- Route-only components (used by a single `+page.svelte`) are co-located next to that page, not in `src/lib/`
- Imports: `$lib/` absolute paths for cross-boundary refs, `./` relative for co-located files

## Store Modules

| File | Purpose |
|------|---------|
| `datasetImages.ts` | Types (DatasetInfo, ImageInfo, ImagePage, CaptionData) + API helpers + deleteDataset + createDatasetBrowserStore |
| `envs.ts` | Env types + CRUD + model fetching |
| `templates.ts` | Template types + CRUD + `extractVariables()` |
| `captionOptions.ts` | `CaptionOptions` type (mirrors `CaptionJobOptions`) |
| `configs.ts` | Config CRUD + export API |
| `events.ts` | Self-connecting SSE store — opens `TypedEventSource` on module load (browser), validates with Zod, pipes into `readonly` writable stores. Exports `captioningStatus`, `pendingDatasetChanges`, `resumptionFailed`, `clearPendingDatasetChange()`, `clearResumptionFailed()`. Uses browser's built-in `EventSource` auto-reconnect (preserves `Last-Event-ID`). |
| `toasts.ts` | Toast notification store — manages a reactive list of active toasts with auto-dismiss. Exports `toasts` readable store, `addToast()`, `dismissToast()`, and `toast.success/error/warning/info()` convenience helpers. |
| `captioning.ts` | Re-export shim from `events.ts` for backward compatibility |
| `captionSettings.ts` | Last-used caption settings persisted to localStorage (env, maxTokens, imageQuality, etc.) — restored on panel open, saved on "Start Captioning". Priority: localStorage override → dataset config default → hardcoded default. |
| `settings.ts` | UI settings (localStorage) — `notifications` tri-state (`"unset"` / `"enabled"` / `"disabled"`) for browser notification preference |

## Key Patterns

- **Hash routing**: SvelteKit uses `router: { type: "hash" }` — all internal links use `#/` prefix. Flask only serves `GET /` + static assets.
- **SSE**: `stores/events.ts` is a self-connecting store module. Opens `TypedEventSource` on module load, validates events with Zod, pipes into `readonly` writable stores. The browser's built-in `EventSource` auto-reconnect handles reconnection automatically — it preserves and sends `Last-Event-ID` on reconnect, allowing the backend to replay missed events from its ring buffer. If resumption fails (history too old), a `resumption_failed` event sets a `resumptionFailed` store, and a warning toast is fired (plus inline banner on the dataset detail page). The `onerror` handler deliberately does **not** call `close()` (which would discard the last-event-id state). A 5s interval fallback reconnect handles edge cases where the EventSource ends up in CLOSED state.
- **Dataset watcher**: Backend emits `DatasetChangedEvent` via SSE when filesystem changes are detected. Frontend stores these in `pendingDatasetChanges` (a `Set<string>`). Dataset browser page subscribes and shows a "Refresh" banner.
- **CodeMirror 6**: `CodeMirror.svelte` wrapper uses three separate `$effect` blocks (create/destroy/sync) — never combine. Uses `editable` prop (default `true`) — not `readonly`. Includes a `baseTheme` (dark surface, accent-colored selection via `color-mix(in oklch, ...)`) and a `darkHighlightStyle` that maps all `@lezer/highlight` tags to CSS `--color-syn-*` variables defined in the Tailwind `@theme` block.
- **Svelte 5**: No pipe directives on events. No nested `<button>`. Use `<div role="button">` for clickable list items.
- **Tailwind v4**: Custom colors must be registered in `@theme { }` block, not `:root` vars.
- **Tailwind content detection**: Root `.gitignore` `lib/` rule was hiding `src/lib/` — fixed with `!src/lib/` in `webui/.gitignore`.
- **npm security**: `min-release-age=14` in `.npmrc` blocks installing packages published <14 days ago.
- **Z-index layers**: Fixed-position elements use a consistent z-index stack:
  - `z-10`: Local absolute-positioned overlays within components (e.g. ImageDetail hover/delete masks)
  - `z-20`: FABs (mobile caption settings floating button)
  - `z-30`: Mobile overlay backdrops (side panel scrim)
  - `z-40`: Mobile slide-in panels (side panel drawer on small screens)
  - `z-50`: Global overlays — dialogs (`Dialog.svelte`) and toast stack (`ToastContainer.svelte`)
  - When adding new fixed/absolute layers, use the appropriate slot and avoid values outside this scale.
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
