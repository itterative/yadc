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
        dataset/                        # Dataset-domain components
          DatasetImage.svelte         # Masonry grid tile (thumbnail + badges + selected outline)
          DatasetBrowser.svelte       # Masonry grid container (column distribution + infinite scroll + selectedId)
          ImageDetail.svelte          # Image detail side panel (full image + caption edit + TOML viewer + drafts + PromptPreview)
          CaptionProgress.svelte      # Captioning progress display with SSE status stream
        dialogs/                        # Dialog-shaped components
          EnvManager.svelte           # Environment CRUD dialog
          ExportDialog.svelte         # Export dialog (backend + draft/caption source selection)
          SettingsDialog.svelte       # App settings dialog (ConfigEditor + TemplateManager + EnvManager launcher)
        settings/                       # Settings-domain sub-components
          EnvSelector.svelte          # Environment form (env dropdown + URL/token/model, bindable props, reload trigger)
          ConfigEditor.svelte         # Full config CRUD panel (sidebar list + TOML editor + save/delete)
          TemplateManager.svelte      # Full template CRUD panel (sidebar list + Jinja editor + new/save/delete)
      icons/             # SVG icon components (SvgClose, SvgDelete, SvgEdit, SvgFile, SvgImage, SvgLogout, SvgPlus, SvgRefresh, SvgSpinner)
    routes/
      layout.css        # Tailwind v4 imports + @source workaround + dark theme
      +layout.svelte    # App shell with breadcrumb nav (hash routing links)
      +page.svelte      # Dataset listing (cards with edit/delete, add-dataset dashed card) → links to #/datasets/{name}
      AddDatasetDialog.svelte   # Co-located: create/import dataset dialog (used only by +page.svelte)
      EditDatasetDialog.svelte  # Co-located: edit dataset TOML config dialog (CodeMirror TOML editor)
      datasets/[name]/
        +page.svelte              # Dataset browser (masonry grid + side panel)
        CaptionSettings.svelte    # Co-located: captioning settings side panel
        SidePanel.svelte          # Co-located: tabbed side panel (caption/details) with mobile drawer
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
| `events.ts` | Self-connecting SSE store — opens `TypedEventSource` on module load (browser), validates with Zod, pipes into `readonly` writable stores. Exports `captioningStatus`, `pendingDatasetChanges`, `clearPendingDatasetChange()` |
| `captioning.ts` | Re-export shim from `events.ts` for backward compatibility |
| `settings.ts` | UI settings (localStorage) |

## Key Patterns

- **Hash routing**: SvelteKit uses `router: { type: "hash" }` — all internal links use `#/` prefix. Flask only serves `GET /` + static assets.
- **SSE**: `stores/events.ts` is a self-connecting store module. Opens `TypedEventSource` on module load, validates events with Zod, pipes into `readonly` writable stores. Components import stores directly — no SSE connection logic in page components. Per-dataset SSE (e.g. `CaptionProgress.svelte`) creates its own `TypedEventSource`.
- **Dataset watcher**: Backend emits `DatasetChangedEvent` via SSE when filesystem changes are detected. Frontend stores these in `pendingDatasetChanges` (a `Set<string>`). Dataset browser page subscribes and shows a "Refresh" banner.
- **CodeMirror 6**: `CodeMirror.svelte` wrapper uses three separate `$effect` blocks (create/destroy/sync) — never combine. Uses `editable` prop (default `true`) — not `readonly`. Includes a `baseTheme` (dark surface, accent-colored selection via `color-mix(in oklch, ...)`) and a `darkHighlightStyle` that maps all `@lezer/highlight` tags to CSS `--color-syn-*` variables defined in the Tailwind `@theme` block.
- **Svelte 5**: No pipe directives on events. No nested `<button>`. Use `<div role="button">` for clickable list items.
- **Tailwind v4**: Custom colors must be registered in `@theme { }` block, not `:root` vars.
- **Tailwind content detection**: Root `.gitignore` `lib/` rule was hiding `src/lib/` — fixed with `!src/lib/` in `webui/.gitignore`.
- **npm security**: `min-release-age=14` in `.npmrc` blocks installing packages published <14 days ago.
