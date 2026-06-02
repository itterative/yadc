---
name: webui-frontend
description: yadc webui frontend setup — SvelteKit hash routing, Tailwind v4 configuration, Flask integration, and known issues.
---

# WebUI Frontend Setup

## Routing: Hash-Based SPA

The webui is served by Flask as a static SPA. With pathname routing (`/datasets/foo`), Flask needs route hacks to serve `index.html` for every possible path. Hash routing (`#/datasets/foo`) means Flask only serves `index.html` for `/` — all navigation is client-side via the URL hash fragment.

### Configuration

**`svelte.config.js`:**
```js
kit: {
  adapter: adapter(),
  router: { type: "hash" },
}
```

- `fallback: "index.html"` is **not** needed
- `+layout.ts` with `export const prerender = true; export const ssr = false;` is **not** needed — hash routing disables both automatically
- No `handleUnseenRoutes` config needed

### Link Format

All internal navigation links **must** use `#/` prefix:

```svelte
<!-- ✅ Correct -->
<a href="#/">Home</a>
<a href="#/datasets/{name}">{name}</a>

<!-- ❌ Wrong — will full-page reload or 404 -->
<a href="/">Home</a>
<a href="/datasets/{name}">{name}</a>
```

SvelteKit's `<a href>` with hash routing does **not** automatically transform `/foo` into `#/foo` — you must write the hash prefix explicitly.

### Flask Side

Flask only needs:
- `GET /` → serve `index.html`
- `GET /_app/<path>` → serve static assets
- No 404 fallback or path-based SPA routes needed

### Build Output

Produces a flat `build/` directory with `index.html`, `robots.txt`, and `_app/` — no per-route HTML files.

## Tailwind CSS v4

### Content Detection — RESOLVED

**Root cause**: Two issues, both now fixed:

1. **`.gitignore` `lib/` rule** — The root `.gitignore` has a blanket `lib/` pattern (for Python packaging artifacts) that matched `yadc/webui/src/lib/`. Tailwind's `@tailwindcss/vite` uses git's tracked file list for content discovery, so `src/lib/` was invisible. **Fix**: Added `!src/lib/` negation in `yadc/webui/.gitignore`.

2. **`:root` instead of `@theme`** — Custom colors were defined as plain CSS variables in `:root { --color-bg: ... }`. Tailwind v4 needs them in `@theme { }` to register them as theme values and generate utilities like `bg-surface`, `text-accent`, etc. The reference project uses `@theme` correctly. **Fix**: Changed `:root` block to `@theme` block in `layout.css`.

No `@source` directives are needed.

## CSS Architecture

- `layout.css` is the Tailwind entry point, imported by `+layout.svelte`
- Custom theme colors registered via `@theme { }` block (Tokyo Night dark palette + `--color-syn-*` variables for CodeMirror syntax highlighting)
- Component classes are split into domain-specific CSS files under `src/lib/styles/`:
  - `buttons.css` — `.btn`, `.btn-primary`, `.btn-secondary`, `.btn-danger`, `.btn-close`
  - `forms.css` — `.input`, `.input-sm`, `.label`, `.help-text`
  - `overlays.css` — `.dialog-panel`, `.dialog-header`, `.dialog-title`, `.alert-error`, `.alert-success`
  - `badges.css` — `.badge`, `.badge-accent`, `.badge-success`, `.badge-error`, `.badge-muted`
  - `utilities.css` — `.btn-bar`, `.section-heading`, `.loading-center`, `.empty-state`, `.card`, `.card-body`, `.dot-separator`
- All component class files use `@layer components { }` — imported via `@import '$lib/styles/...'` in `layout.css`
- Component-specific styles use Svelte's `<style>` scoped blocks (e.g. `.grid-cols-auto` in `DatasetBrowser.svelte`, nav styles in `+layout.svelte`)

### Vite Plugin Order

`tailwindcss()` before `sveltekit()` in `vite.config.ts`:

```ts
plugins: [tailwindcss(), sveltekit()],
```
