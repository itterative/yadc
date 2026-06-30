---
name: frontend-architecture
description: WebUI frontend conventions — component and store organization rules, styling patterns. Read architecture-overview first if touching backend too.
category: architecture
priority: 2
keep_updated: true
---

# Frontend Architecture

**Stack**: SvelteKit (Svelte 5 + runes) + Tailwind CSS v4 + TypeScript + Zod

**Backend reference**: See `architecture-overview` for the Quart API backend.

## See also (in `.pi/agent/memory/docs/`)

- `frontend/` — file-by-file structure (one file per area, each with one-line summaries per file)
- `frontend-patterns` — frontend patterns (Tabs, Z-index, Topbar, Browser notifications, Drop-to-upload, SSE)

## Component Organization (in `yadc/webui/src/`)

**Where a component lives** is decided by two rules:

### Scope rule (placement)

1. **`lib/components/ui/`** — atomic UI primitives, no domain concepts, no API calls.
2. **`lib/components/<domain>/`** — domain components, used in ≥2 places OR used by a `lib/` component.
3. **`routes/<path>/`** (co-located) — used by exactly one route AND not by anything in `lib/`. **Promote to `lib/` on the second consumer.**

### Feature-folder rule (when a feature gets its own sub-folder)

A feature gets its own sub-folder when it has ≥3 related `.svelte` files forming a coherent unit (a host component + its tabs/sections), all of which are used only by that host. Folder name = feature name. The host file shares the folder name. Inside the folder:

- `Feature.svelte` — the host
- `FeatureTab.svelte` — tab children
- `state.svelte.ts` — shared runes (if any)
- `helpers.ts` — pure helpers used by multiple files in the folder

Pure helpers used **outside** the folder stay in `lib/` proper.

### Imports

- `$lib/` absolute paths for cross-boundary refs.
- `./` relative for co-located files.
- No file in `lib/` imports from `routes/` — preserved by the "promote on second use" rule.

### Current layout (one-liner per folder)

- `lib/components/ui/` — atomic primitives: `Dialog`, `Alert`, tabs system (`Tabs`/`Tab`/`PillTabs`/`CompactPillTabs`), `CodeMirror`, `SpinnerBlock`, `SparkleThinking`, `ToastContainer`/`ToastItem`, `Tooltip`, `FileDropZone`, `Checkbox`, `Slider` (native range input, themed via `accent-color`), `PromptPreview`, `Card`, `ActionBar`/`ActionBarItem`, `IntersectionObserverElement`, `ContextMenu`, `ImagePreviewDialog`.
- `lib/components/dataset/` — dataset domain. Sub-folders per feature: `browser/`, `detail/`, `config/`, `upload/`, `manage/`.
- `lib/components/caption/` — `CaptionSettingsPanel` (the batch-captioning side panel).
- `lib/components/tagging/` — `TagSettingsPanel` (the batch-tagging side panel, mirrors `caption/CaptionSettingsPanel`).
- `lib/components/env/` — `EnvironmentSettings` (CRUD list), `EnvSelector` (caption-flow picker).
- `lib/components/export/` — `ExportDialog`.
- `lib/components/settings/` — `SettingsDialog` (host) + tabs (`GeneralSettings`/`SecuritySettings`) + `CaptionOptionsFields` widget. `TemplateManager.svelte` is **marked LEGACY** — superseded by `/templates` route.
- `lib/components/templates/` — `EditTemplateDialog` (used by both the `/templates` route and the caption tab sidebar).
- `lib/components/dialogs/` — only `PasswordPromptDialog` (global utility).
- `lib/icons/` — `Svg*` SVG components (`viewBox="0 -960 960 960"`, sized/colored via Tailwind)
- `lib/actions/` — Svelte actions. `autosize.ts` (textarea auto-resize, capped at `maxHeight`) and `autoscroll.ts` (follow streaming content — attaches to the element whose content streams, but scrolls the element that actually scrolls that content: itself if it's a scroll container, else the nearest scrollable ancestor, else the page. This lets it drive an inner scroll container like `ReasoningCard`'s `<pre>` AND an area-level scroll where the streaming element has no own overflow, e.g. the prompt preview on mobile where the whole content area scrolls together. Re-resolves the target on viewport resize so a responsive scroll switch (inner container desktop ↔ content area mobile) re-targets. Scrolls to bottom on mount, then on every content change until the user scrolls up; auto-scroll resumes only when the user scrolls back to the bottom. `scrollOnMount` option. Replaces an earlier position-threshold model that required users to scroll aggressively up to disable).

Note: `lib/components/dialogs/` and `lib/icons/` are pragmatic exceptions to the Scope rule above — `dialogs/` holds global utilities (not a domain) and `icons/` lives outside `lib/components/` entirely.

Route-only components are co-located in `routes/<path>/`. Full listing in `frontend/routes.md`.

## Store Organization (in `yadc/webui/src/lib/stores/`)

Mirrors the component organization conventions.

### Scope rule (placement)

1. **`lib/stores/<domain>/`** — stores serving a single domain. ≥2 related files (types + API + store + actions), all serving the same domain. Each sub-folder has a re-exporting `index.ts` so callers import from `$lib/stores/<domain>`.
2. **`lib/stores/`** (top-level) — global UI primitives (toasts, confirm, password, session password, settings, topbar) and the SSE event backbone. Singletons that don't form a coherent feature.

### Sub-folder rule

A domain gets its own sub-folder when it has ≥2 related files AND they all serve the same domain. Inside: `index.ts` (re-exporting barrel), `types.ts` (data shapes), `api.ts` (transport — fetch/CRUD), plus `store.ts`/`actions.ts` when reactive state is involved, plus pure helpers (e.g. `jinja.ts`).

### Current layout

- 6 domain sub-folders: `dataset/` (4 files), `caption/` (9 files), `tagging/` (10 files), `config/` (3 files), `env/` (3 files), `templates/` (4 files).
- 7 top-level singletons: `events.ts`, `toasts.ts`, `confirm.ts`, `passwordPrompt.ts`, `sessionPassword.ts`, `settings.ts`, `topbar.svelte.ts`, `storageStore.ts`.

Full per-file summary in `frontend/stores.md`.

### Naming convention for sub-folder files

Two styles coexist; use whichever fits the domain's size:

- **Role-based** (used by `dataset/`, `config/`, `env/`, `templates/`): one file per kind of code — `types.ts` (data shapes), `api.ts` (transport/fetch), `store.ts` (writables + actions), `actions.ts` (higher-level operations), `index.ts` (re-exporting barrel), plus domain-specific helpers (e.g. `jinja.ts`).
- **Feature-based** (used by `caption/` once it grew past 3-4 files): one file per **concern** within the domain — `status.ts`, `inflight.ts`, `timing.ts`, `jobs.ts`, `refined.ts`. Each file owns the store, the actions, and the types for that one concern.

Use role-based while the domain is small; switch to feature-based when adding a new file would feel forced under the role taxonomy (e.g. another `store.ts` in the same folder is a smell).

### `events.ts` is the router, not the store

`lib/stores/events.ts` is intentionally top-level (not a sub-folder) and owns only the SSE transport + event-routing layer. Domain state lives in the relevant domain sub-folder. The rule:

- `events.ts` defines the Zod schemas for every event type
- `events.ts` defines the `TypedEventSource` lifecycle (connect, reconnect, mobile visibility recovery, fallback interval)
- `events.ts` owns `clientId` (tab identity for `dataset_changed` suppression)
- `events.ts` owns the SSE-specific stores that don't fit any domain: `resumptionFailed`, `lastCaptionedImage`, `lastCaptionError`
- Everything else — `captioningStatuses`, `currentlyCaptioning`, `captionTimingRing`, `imageRefined`, `pendingDatasetChanges`, `activeJobIds` — lives in the relevant domain sub-folder and is written via a domain-owned action (e.g. `setCaptioningStatus`, `addCurrentlyCaptioning`, `recordCaptionTiming`, `setImageRefined`, `addPendingDatasetChange`, `isOwnJobId`).

This keeps `events.ts` small and means each store can be reasoned about (and tested) in isolation — the SSE handler is just a dispatch layer, not a god module.

### Imports

- `$lib/stores/<domain>` (re-exporting barrel) for cross-boundary refs.
- `./<file>` relative for files inside the same sub-folder.
- No file in `lib/` imports from `routes/`.

## Styling Patterns

Three-tier approach for mixing Tailwind utilities, reusable component classes, and scoped Svelte styles:

### 1. `@layer components` in CSS files (`yadc/webui/src/lib/styles/`)

For **reusable UI abstractions** that appear across multiple components. Use `@apply` to compose Tailwind utilities into named classes.

- `buttons.css` — `.btn`, `.btn-primary`, `.btn-secondary`
- `forms.css` — `.input`, `.label`
- `badges.css` — `.badge`, `.badge-sm`, `.badge-accent`
- `utilities.css` — `.diff-dot`, `.card`, `.section-heading`
- `overlays.css` — `.dialog-panel`, `.alert-{error,warning,success,info}`

**Rules**:
- Extract appearance only (colors, typography, border-radius, padding). **Do not include positioning** (absolute, fixed, margins for layout) — that stays inline.
- Keep semantic names that describe what the thing *is*, not where it sits.

### 2. Inline Tailwind utilities in HTML

For **one-off structural and layout styles** specific to a single element. Examples: `class="h-full overflow-hidden"`, `class="min-w-0 overflow-hidden"`, `class="max-sm:gap-3 max-sm:px-4"`, `class="absolute top-1 right-1"`.

### 3. Svelte `<style>` scoped blocks

For styles that genuinely can't be utilities:

- **State classes** applied via `class:selected` or similar — e.g. `.selected { outline: 2px solid var(--color-accent); }`
- **JS-driven CSS variables** — e.g. `.grid-cols-auto { grid-template-columns: repeat(var(--x-grid-cols), minmax(0, 1fr)); }` where `--x-grid-cols` is set via inline style from JS
- **Complex structural layout** that's easier to read as CSS — e.g. the navbar layout in `+layout.svelte`

**Anti-patterns to avoid**:
- Using `@apply` inside Svelte `<style>` blocks — adds indirection without benefit. Either extract to `lib/styles/` (if reusable) or inline utilities (if one-off).
- Creating component classes that include positioning (e.g. `.badge-corner` with `absolute`). Positioning is context-specific and should be inline.
- Writing raw CSS for styles that map directly to existing utilities (e.g. `display: flex; align-items: center;` instead of `class="flex items-center"`).
