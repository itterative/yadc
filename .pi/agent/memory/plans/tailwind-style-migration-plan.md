---
name: tailwind-style-migration-plan
description: Migrate remaining manual CSS in Svelte components to Tailwind utilities, keeping the three-tier styling architecture intact.
last_history: 0
---

# Tailwind Style Migration Plan

## Current State

The webui follows a three-tier styling architecture (see `frontend-architecture` memory):
1. `@layer components` in `src/lib/styles/*.css` for reusable abstractions (`.btn-primary`, `.input`, `.badge`)
2. Inline Tailwind utilities in HTML for one-off structural/layout styles
3. Svelte `<style>` scoped blocks only when genuinely necessary (state classes, JS-driven CSS vars, complex layout)

There are **4 remaining scoped `<style>` blocks** and a handful of inline `style=` attributes that violate this architecture.

## Files Requiring Changes

### Scoped `<style>` blocks (4 files)

| File | Lines | What it does | Migration difficulty |
|------|-------|--------------|----------------------|
| `routes/+layout.svelte` | ~180 | App shell: sidebar (mobile overlay + desktop rail), topbar, burger button, nav links, brand, backdrop scrim. Uses `@media (min-width: 768px)`, `:empty`, `:global(.nav-icon)`. | High — most complex responsive layout in the app; markup will become very dense with utilities. |
| `lib/components/dataset/DatasetImage.svelte` | ~50 | `.selected` outline, `.flashing`/`@keyframes tile-flash`, `.captioning` shimmer/`@keyframes shimmer`, interaction rule `.captioning.flashing`. | Medium — animations need a home; rest is state-class toggling. |
| `lib/components/dataset/DatasetBrowser.svelte` | ~7 | `.grid-cols-auto` with CSS custom property `--x-grid-cols` set from JS. | Low — can become arbitrary Tailwind value or stay as inline style. |
| `lib/components/ui/Tooltip.svelte` | ~35 | Pure-CSS hover tooltip: absolute positioning, opacity transition, mobile `@media` hide. | Low — almost entirely replaceable with `group-hover`, `max-md:hidden`, etc. |

### Global base styles in `routes/layout.css`

Outside the `@theme` block:
- `body { background, color, font-family, margin }`
- `a / a:hover` colors
- `body.noscroll { overflow: hidden }` (toggled by `Dialog.svelte`)

These are acceptable as global base styles, but could be moved to `@layer base` if desired.

### Inline `style=` attributes (3 files)

| File | Attribute | Replace with |
|------|-----------|--------------|
| `routes/templates/EditTemplateDialog.svelte` | `style="min-height: 200px;"` | `style="min-height: 200px"` or Tailwind arbitrary `min-h-[200px]` |
| `lib/components/dataset/DatasetBrowser.svelte` | `style="overflow: hidden;"` | `class="overflow-hidden"` (Tailwind utility) |
| `lib/components/settings/TemplateManager.svelte` | `style="height: 50vh;"` | `class="h-[50vh]"` (Tailwind arbitrary) |

### Reactive `style:` directives (keep as-is)

These are the correct tool for JS-driven values and should **not** be migrated:
- `ToastItem.svelte`: `style:width="{pctRemaining}%"`
- `routes/datasets/[name]/+page.svelte`: `style:width="{captionPct}%"`

## Approach Options

### Option A: "Surgical" — Scoped `<style>` only
Refactor the 4 Svelte files with `<style>` blocks. Leave everything else untouched.
- Tooltip → pure Tailwind utilities (`group`, `absolute`, `group-hover:opacity-100`, `max-md:hidden`)
- DatasetBrowser grid → Tailwind arbitrary value `grid-cols-[repeat(var(--x-grid-cols),minmax(0,1fr))]` (CSS var still from JS inline)
- DatasetImage → mixed; keep tiny `<style>` block just for keyframes, or move keyframes global
- `+layout.svelte` → biggest lift; every structural rule becomes inline utilities

**Verdict:** acceptable but leaves `+layout.svelte` as a large outlier.

### Option B: "Full purge" — Eliminate all manual CSS
Also migrate `src/lib/styles/` `@apply` files into inline Tailwind on components, and move global base styles into `@layer base`.

**Verdict:** rejected — `.btn-primary`, `.input`, `.badge` are used in ~20+ places; inlining creates duplication and makes global changes painful. This violates the project's stated anti-pattern (see `frontend-architecture` memory).

### Option C: "Keep the architecture, clean the edges" (preferred)
- Migrate the 4 scoped `<style>` blocks where feasible.
- Keep `src/lib/styles/` `@layer components` as-is — already Tailwind-based and working well.
- Keep global base styles in `layout.css` — they belong there.
- Replace the 3 inline `style=` attributes with Tailwind utilities.
- Keep the `style:` directives — correct tool for reactive JS-driven values.

### Option D: "Hybrid animation strategy" (preferred, for DatasetImage)
For `DatasetImage.svelte` specifically:
- Move `@keyframes tile-flash` and `@keyframes shimmer` into `layout.css` under `@theme` or a new `@layer utilities` block (Tailwind v4 `@keyframes` registration).
- Use Tailwind utilities like `animate-[tile-flash_0.8s_ease-out]` and `animate-[shimmer_3.5s_ease-in-out_infinite]` for class toggling.
- This deletes almost the entire `<style>` block while keeping custom animations defined once globally.

## Recommended Approach

**Combine C + D.**

1. **Tooltip** — migrate entirely to Tailwind utilities (no `<style>` left).
2. **DatasetBrowser** — replace `.grid-cols-auto` with `class="grid grid-cols-[repeat(var(--x-grid-cols),minmax(0,1fr))]"`; remove `<style>`.
3. **DatasetImage** — move keyframes to `layout.css`, replace classes with `animate-[...]` utilities; remove `<style>`.
4. **+layout.svelte** — migrate structural/layout rules to inline Tailwind. Keep `:global(.nav-icon)` as the only `:global()` rule. The `:empty` rule becomes `empty:hidden` utility. This is the largest file but aligns it with project conventions.
5. **Inline `style=` cleanup** — replace the 3 occurrences with Tailwind utilities.
6. **Do NOT touch** — `src/lib/styles/*.css`, `style:` directives, CodeMirror baseTheme (CM6 theme object, not CSS).

## Open Decisions

- Should `+layout.svelte` be fully Tailwind-ified, or is its complexity large enough that a reduced `<style>` block is acceptable for readability?
- Where exactly should custom keyframes live in Tailwind v4? (`@theme` block vs `@layer utilities` vs a separate `@import` file)

## Acceptance Criteria

- [ ] Zero scoped `<style>` blocks remaining in `lib/components/ui/Tooltip.svelte`
- [ ] Zero scoped `<style>` blocks remaining in `lib/components/dataset/DatasetBrowser.svelte`
- [ ] Zero scoped `<style>` blocks remaining in `lib/components/dataset/DatasetImage.svelte`
- [ ] `routes/+layout.svelte` scoped styles reduced to only what is impractical in Tailwind (or eliminated entirely)
- [ ] No inline `style=` attributes in `.svelte` files (except reactive `style:` directives)
- [ ] `src/lib/styles/` files untouched
- [ ] `npm run build` passes, `npx svelte-check` passes
- [ ] Visual parity confirmed for: sidebar responsive behavior, tooltip hover, dataset grid layout, image selection/shimmer/flash animations
