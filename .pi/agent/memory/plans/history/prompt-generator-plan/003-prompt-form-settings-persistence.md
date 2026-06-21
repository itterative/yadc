---
date: 2026-06-20
---
# Prompt-form settings persistence (localStorage)

**Context:** The prompt generator page (`/prompts`) had a useability
gap: the small text fields (env, api url/token/model, intent, focus)
reset to defaults on every page reload. Re-typing a long intent after
a refresh was annoying. The user asked for a storable to persist these
settings across reloads, and asked for my opinion on whether to also
persist the few-shot `examples` (which carry base64 image data URLs).

**Decision:** Persist the small text fields via localStorage in a
new `promptSettings` store (`$lib/stores/prompts/settings.svelte.ts`).
Drop the few-shot `examples` from persistence entirely (they live in
memory only).

**Why drop images for now:**

- A single 1MB JPEG becomes ~1.3MB as a base64 data URL (× 1.33
  overhead). A handful of examples blows past localStorage's ~5–10MB
  quota, and `setItem` throws `QuotaExceededError` on overflow.
- Even when it fits, sync localStorage writes block the main thread
  for non-trivial blobs — visible jank on long inputs.
- We don't currently track per-example provenance (e.g.
  `{type: "dataset", dataset, imageId}`) that would let us re-fetch
  dataset-sourced images on load. Adding that is a real refactor
  (touch the `ExamplePair` model, the `ExamplesPanel` add modes, and
  the wire format) and was deferred.

**Implementation:**

- `promptSettings` is created via the project's shared `storable`
  utility (`$lib/storable.js`) — the same pattern used by
  `stores/settings.ts` (`yadc/settings`) and
  `stores/caption/settings.ts` (`yadc/captionSettings`). Reuses
  the JSDoc-typed `storable<T>(key, defaults, migrate?)` factory
  that already handles localStorage read/write, a no-op SSR
  fallback (`typeof window === 'undefined'`), `$version` for
  future shape migrations, and a `console.error` on parse failure
  (falls back to defaults). Storage key: `yadc/prompts/formSettings`
  (`$version: 1`).
- `resetPromptSettings()` does `promptSettings.set({...defaults})`
  — the underlying `storable` subscribe handler persists the
  reset, so the localStorage entry is updated without a separate
  `removeItem` call.
- **The host component (`PromptGenerator.svelte`) uses the
  controlled-component pattern**, NOT `bind:prop={$store.field}`:
  local `$state` for env / apiUrl / apiToken / apiModelName /
  intent / focus is the source of truth for the form, initialized
  once from `$promptSettings` at mount, and a single `$effect`
  mirrors every change back to the store via `promptSettings.update`.
  The `bind:prop` shorthand in `PromptForm` then binds cleanly to
  local `$state` (no store-property write involved). This mirrors
  the pattern in `GeneralSettings.svelte` (which syncs to
  `yadc/settings` via `settings.update`).
- **Why not `bind:prop={$store.field}`?** Svelte 5's `$store`
  auto-subscription is read-only — writing to a property of the
  auto-subscribed value doesn't propagate back to the store, so
  the form's writes silently no-op against persistence. Worse,
  downstream consumers (like `PromptForm`'s env-change `$effect`)
  that watch store-side state can end up in an
  `effect_update_depth_exceeded` loop because the read sees the
  pre-write snapshot while other reactive paths keep firing.
  Two-way binding into a store wants either the explicit
  function getter/setter form
  (`bind:value={() => $store.field, (v) => store.update(...)}`)
  or the controlled-component pattern we went with.
- `examples` stays as a local `$state<ExamplePair[]>` (the
  dropped-on-reload list).
- `handleGenerate` snapshots the local form values at click time
  so the streaming run uses them even if the user edits the form
  mid-stream.

**Notes for future me:**

- The env-change effect in `PromptForm` auto-fills
  `apiUrl`/`apiModelName` from the env config. With persistence,
  switching envs now overwrites any custom values the user had
  stored. The user accepted this trade-off (the existing
  auto-fill behavior was already that way, and re-typing custom
  values is a minor inconvenience vs. the larger win of having
  intent + env persist).
- The pattern is consistent with the rest of the app: persisted
  settings use `storable` (writable store), ephemeral state uses
  Svelte 5 runes (`$state`). The host component bridges the two
  via the controlled-component pattern — local `$state` is the
  form's source of truth, a single `$effect` syncs to the store.
- **If you find yourself reaching for `bind:prop={$store.field}`**
  anywhere in the codebase: don't. Use the controlled-component
  pattern or the explicit function getter/setter form.

## Files

### New
- `yadc/webui/src/lib/stores/prompts/settings.ts` —
  `promptSettings` Svelte writable store (returned by
  `storable<PromptFormSettings>`) and `resetPromptSettings()`.
  Storage key `yadc/prompts/formSettings`, `$version: 1`. Plain
  `.ts` (not `.svelte.ts`) because it doesn't use runes — the
  settings persistence is delegated to the `storable` utility,
  matching `stores/settings.ts` and `stores/caption/settings.ts`.

### Updated
- `yadc/webui/src/lib/stores/prompts/index.ts` — barrel
  re-exports `settings`.
- `yadc/webui/src/lib/components/prompts/PromptGenerator.svelte`
  — keeps the original local `$state` for env/apiUrl/apiToken/
  apiModelName/intent/focus (initialized once from
  `$promptSettings` at mount) and adds a single `$effect` that
  syncs every change back to the store via `promptSettings.update`.
  Bindings in `PromptForm` use plain `bind:prop` against the local
  `$state` (no store-property writes). `examples` stays as local
  state.
- `.pi/agent/memory/docs/frontend/stores.md` — documented the
  new `settings.ts` in the `prompts/` sub-folder listing, plus
  a note that `bind:prop={$store.field}` doesn't work in Svelte 5.

## Deferred

- **IndexedDB for example image data URLs.** localStorage's
  ~5–10MB ceiling and sync-write overhead make it a bad fit for
  arbitrary image blobs. IndexedDB is async, has orders-of-magnitude
  more capacity (typically half of free disk), and is the right
  tool for this. The migration shape would be: add a `source` field
  to `ExamplePair` (`{type: "manual"}` or
  `{type: "dataset", dataset: string, imageId: string}`), and
  re-fetch dataset-sourced images on page load; manual uploads
  still need to be re-added (or also stored in IDB). Worth doing
  as a focused follow-up; tracked in `todo.md`.
