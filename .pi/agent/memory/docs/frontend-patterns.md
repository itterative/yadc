---
name: frontend-patterns
description: WebUI frontend patterns — Abort contexts, Tabs system, Z-index layers, Topbar pattern, Browser notifications, Svelte 5 conventions, Drop-to-upload, SSE.
category: architecture
keep_updated: true
---

# Frontend Patterns

Patterns specific to the yadc webui frontend. For SvelteKit/Tailwind v4 setup see `webui-frontend`. For the component organization rules (placement, feature-folder rule) and store organization rules, see `frontend-architecture`. For CodeMirror sizing pitfalls, see `codemirror-quirks`. For the inotify event payload and suppression, see `dataset-watcher`.

## See also (in `.pi/agent/memory/docs/`)

- `webui-frontend` — SvelteKit/Tailwind v4 setup
- `codemirror-quirks` — CodeMirror sizing pitfalls
- `dataset-watcher` — Backend ring buffer + suppression pattern (SSE)

## Tabs System

`ui/tabs/` uses Svelte context for parent-child coordination.

- **`TabsContext.svelte.ts`** — Symbol key + state factory + typed helpers. `TabItem` has optional `icon` (Svelte `Component<{ class?: string }>`).
- **`Tabs.svelte`** — Generic container. Tab bar layout, registration, bindable `value`. Supports a custom `tab` snippet for arbitrary button styling.
- **`PillTabs.svelte`** — Pre-styled variant (rounded-full buttons on border-bottom bar). Wraps `Tabs`.
- **`CompactPillTabs.svelte`** — Compact segmented-control variant (`bg-gray-800/50` container, rounded-md buttons) with icon support. Uses `TabsContext` directly rather than wrapping `Tabs`.
- **`Tab.svelte`** — Child that auto-registers via context during init via `untrack()` (synchronous, before parent renders). Shows/hides its slot content. Accepts optional `icon` prop.

`value` is bindable and uses `id` (not `label`) for matching.

## Z-Index Layers

Fixed-position elements use a consistent z-index stack:

| Layer | Use |
|-------|-----|
| `z-10` | Local absolute-positioned overlays within components (e.g. ImageDetail hover/delete masks) |
| `z-20` | FABs (mobile caption settings floating button) |
| `z-30` | Mobile overlay backdrops (side panel scrim, sidebar scrim) |
| `z-40` | Mobile slide-in panels (side panel drawer, sidebar drawer on small screens) |
| `z-50` | Global overlays — dialogs (`Dialog.svelte`) |
| `z-60` | Toast notifications (`ToastContainer.svelte`) — above dialogs so they remain visible |

When adding new fixed/absolute layers, use the appropriate slot and avoid values outside this scale.

## Topbar Pattern

The layout has a topbar (inside `app-content`, between sidebar and main). Pages set topbar content by defining a `{#snippet}` and passing it to `<Topbar>`. The snippet is stored in `topbar.svelte.ts` (a `$state` module). The layout reads it with `getTopbarContent()` and renders with `{@render}`.

- On mobile the topbar also houses the burger menu button.
- On desktop, if no snippet is set, the topbar is hidden via `:empty`.

## Browser Notifications

`notifications.ts` is the single gatekeeper. `sendNotification()` checks support, settings preference (`notifications === "enabled"`), browser permission, and tab visibility — callers just call it with no pre-checks.

- **`promptNotificationsOnce()`** — shows a one-time toast with "Enable" action on first captioning start (session-guarded, only when `notifications === "unset"`).
- Settings dialog General tab has a checkbox that toggles between `"enabled"`/`"disabled"`.
- Global notification dispatching lives in `+layout.svelte` so it works even when the user navigates away from the dataset page.

## Drop-to-Upload in the Dataset Browser

A full-area `DropUploadZone.svelte` wrapper owns the drag/drop state and renders a full-area overlay (accent for allowed drops, warning for blocked). On a valid drop, the page receives the files via `ondrop(files)` callback and opens `AddFilesDialog` pre-populated with the dropped files.

The dialog wraps `DatasetUploadPanel` in `mode="append"`, and the panel's `initialFiles` prop (consumed once via `untrack()`) seeds the file list. An upload icon in the topbar opens the same dialog with no pre-populated files.

Uploads are only allowed when `currentDataset.source === "upload"` and no batch captioning is running — the overlay shows a warning in that case (`uploadBlockedReason` is surfaced via `onblockeddrop(reason)` → toast).

## SSE (Frontend Side)

`stores/events.ts` is a self-connecting store module. Opens `TypedEventSource` on module load, validates events with Zod, pipes into `readonly` writable stores.

- Browser's built-in `EventSource` auto-reconnect handles reconnection automatically — it preserves and sends `Last-Event-ID` on reconnect, allowing the backend to replay missed events from its ring buffer.
- For manual reconnections (e.g. mobile visibility recovery, fallback interval), `TypedEventSource.lastEventId` is tracked and passed as a `?lastEventId=` query parameter.
- A `visibilitychange` listener detects mobile background/foreground transitions: if hidden for > 5 s, the connection is force-closed and reconnected with the tracked event ID.
- The `onerror` handler captures `lastEventId` before the EventSource becomes unusable, but does **not** call `close()` (which would discard the internal last-event-id state).
- A 5s interval fallback reconnect handles edge cases where the EventSource ends up in CLOSED state.
- If resumption fails (history too old), a `resumption_failed` event sets a `resumptionFailed` store, and a warning toast is fired (plus inline banner on the dataset detail page).

See `dataset-watcher` for the backend ring buffer + suppression pattern.

## Svelte 5 Conventions

- No pipe directives on events (`on:click` is gone — use `onclick`).
- No nested `<button>`. Use `<div role="button">` for clickable list items.
- Shared state across components uses `$state` modules (`.svelte.ts` files) — see `topbar.svelte.ts` for the pattern.

## Abort Contexts

`yadc/webui/src/lib/abort.ts` provides a composable abort-context layer for cancelling in-flight fetches when components unmount or effects rerun.

### API

- `createAbortContext()` — creates an `AbortController` linked to the nearest parent context. Children compose with this new signal. Use at route/component level.
- `getAbortContext()` — reads the parent signal (returns `undefined` at the root).
- `linkedController(parent?)` — creates an effect-scoped `AbortController` that aborts when the parent aborts. Use inside `$effect` blocks.
- `setAbortContext(signal)` — used once in `+layout.svelte` to establish the root context.

### Typical pattern in a route or component

```svelte
<script lang="ts">
  import { createAbortContext } from '$lib/abort';

  const abort = createAbortContext();

  $effect(() => {
    loadThings(abort.signal);
    return () => abort.abort();
  });
</script>
```

### Typical pattern inside an `$effect` that reruns on value changes

```svelte
<script lang="ts">
  import { getAbortContext, linkedController } from '$lib/abort';

  const parentSignal = getAbortContext();

  $effect(() => {
    const controller = linkedController(parentSignal);
    fetchThing(id, controller.signal);
    return () => controller.abort();
  });
</script>
```

### API helper convention

Every fetch helper in `yadc/webui/src/lib/stores/<domain>/api.ts` accepts an optional `signal?: AbortSignal` as the last positional parameter and forwards it to `fetch()`. Debounced helpers (`fetchDatasets`, `fetchImages`, `fetchCaption`, `fetchHistory`, `fetchConfig`, `fetchModels`, etc.) use the signal convention in `lib/async.ts`:

- If the last argument is an `AbortSignal`, the debouncer creates its own per-key `AbortController`.
- The callback receives the debouncer's controller signal, not the caller's.
- Callers with the same key share the underlying fetch, but each gets their own promise.
- If a caller's signal aborts, only that caller's promise rejects; the fetch continues for remaining interested callers.
- The fetch is only cancelled when the last interested caller aborts.
- If all callers abort before the debounce timer fires, the timer is dropped and the callback is never invoked.
- The dedupe key ignores the signal so calls with/without signals still dedupe.

### Store refresh helpers

`refreshEnvs()` and `refreshTemplates()` accept a signal and skip updating their stores when aborted, so a late response from a cancelled fetch doesn't clobber newer data.

### Background refreshes

SSE-driven refreshes in `stores/events.ts` intentionally do **not** pass signals. They are global background tasks and should survive page navigation.
