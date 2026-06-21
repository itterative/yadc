---
name: frontend/routes
description: Frontend routes/ — co-located route components (pages + dialogs that are used by exactly one route). The /datasets/[name] route is the most feature-rich.
category: architecture
---

# Frontend: `routes/`

Route components, co-located next to the route that uses them. Per the "scope rule" (in `frontend-architecture`), these are components used by exactly one route AND not by anything in `lib/`. **Promote to `lib/` on the second consumer.**

## See also (in `.pi/agent/memory/docs/`)

- `frontend-patterns` — Drop-to-upload pattern, Topbar pattern
- `webui-frontend` — Hash routing (`#/` prefix)
- `dataset-system` — Upload pipeline

```
yadc/webui/src/routes/
  layout.css        # Tailwind v4 imports + @theme block + typography plugin
  +layout.svelte    # App shell with left sidebar nav (icon-rail on desktop, slide-in overlay on mobile with burger menu). Brand uses android-chrome-192x192.png icon. Tooltip component for desktop hover labels.
  +page.svelte      # Dataset listing (cards with edit/delete, add-dataset dashed card) → links to #/datasets/{name}
  AddDatasetDialog.svelte   # Co-located: dataset creation dialog shell. Two tabs: Upload (via UploadDatasetTab) and Create/Import (via CreateDatasetTab)
  EditDatasetDialog.svelte  # Co-located: edit dataset dialog (Config / Manage / Upload tabs)
  templates/
    +page.svelte              # Template listing (grid cards with edit/duplicate/delete, add-template dashed card) — mirrors dataset listing. Create/edit dialog lives at `$lib/components/templates/EditTemplateDialog.svelte` (reused by the caption tab sidebar). Duplicate flow is a thin frontend wrapper that GETs the source content + PUTs it under the new name — no backend endpoint, since templates are a single file.
    DuplicateTemplateDialog.svelte   # Co-located: prompt for a new template name when duplicating. Warns on name collision (PUT would overwrite).
  prompts/
    +page.svelte              # Prompt generator — hosts `prompts/PromptGenerator.svelte`. Topbar with the page title, full-height two-column layout.
  datasets/[name]/
    +page.svelte              # Dataset browser (masonry grid + side panel + captioning progress in stats line + topbar upload icon)
    SidePanel.svelte          # Co-located: tabbed side panel (caption/details/config) with mobile drawer. Minimal prop threading — captioning actions handled by components via stores.
    AddFilesDialog.svelte     # Co-located: dialog wrapping DatasetUploadPanel in `mode="append"` for adding files to an existing managed dataset. Pre-populates the file list from dropped files.
    DatasetTopbar.svelte      # Co-located: page-level topbar (title, dataset status, action buttons). Uses `SetTopbar` pattern.
    DropUploadZone.svelte     # Co-located: drop-target wrapper with a slot. Owns drag/drop state + `webkitGetAsEntry` file collection. Renders a full-area overlay (accent for allowed drops, warning for blocked). Emits `ondrop(files)` / `onblockeddrop(reason)`. Reused only here today — co-locate until a second consumer appears.
```
