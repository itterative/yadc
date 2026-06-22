---
date: 2026-06-22
---
# Prompt UI: side panel primitive, layout, and two fixes

**Context:** The `/prompts` route still used a two-column CSS-grid
layout (`minmax(0,1fr)_minmax(0,1.2fr)`) that the dataset route had
already abandoned for a slide-over side panel. Beyond the consistency
win, the prompt form was getting tall (env selector, intent, focus,
examples, mode toggle, refine template, footer actions) and the
desktop preview column was cramped. Goal: extract the dataset route's
side panel into a reusable primitive, then move the prompts form into
it. While doing that, two latent bugs surfaced and got fixed.

## Generic `ui/SidePanel.svelte` primitive (extracted)

The dataset route's `SidePanel.svelte` (a `PillTabs` host with a
mobile drawer, FAB toggle, dim backdrop, outside-click-to-close) had
been copy-pasted from a single caller. It became the **generic
primitive** `yadc/webui/src/lib/components/ui/SidePanel.svelte`:

- `children: Snippet` — caller-provided panel content.
- `bind:open` — mobile drawer state (desktop is always inline, the
  binding only affects the drawer).
- `class?: string` — extra Tailwind classes for the panel container
  (intended for width overrides — the panel itself has no default
  width at `lg+`, so the caller sets `lg:w-120` etc.).
- `onclose?: () => void` — fired on outside-click close.
- FAB toggle button (`SvgMenuLeft`) is built into the primitive.
- Outside-click handler is a `<svelte:window onclick>` listener.

The dataset route's wrapper was renamed to
`routes/datasets/[name]/DatasetSidePanel.svelte` (its mobile X close
lives inside the `PillTabs` `end` snippet and mutates `open` via
`bind:open`).

## `PromptSidePanel.svelte` + `PromptGenerator.svelte` refactor

`PromptGenerator.svelte` flipped to the new primitive. Form state
stays in the host (`mode`, `env`, `apiUrl`, `apiToken`,
`apiModelName`, `intent`, `focus`, `maxTokens`, `imageQuality`,
`templateContent`, `examples`) — the panel is a thin
presentational+binding host over them. New
`lib/components/prompts/PromptSidePanel.svelte` wraps the generic
`ui/SidePanel` and hosts the three `PillTabs` tabs (Generate /
Settings / History) + the Generate / Save-to-history footer. The
desktop layout is now: full-page `GenerationPreview` on the left +
`PromptSidePanel` (`lg:static lg:w-120`) on the right. The
`EditTemplateDialog` flow stays in the host (it's opened by the
preview's "Save as template" button — not from the side panel).
`ongenerate` (panel → host) calls `startGeneration` with a **snapshot
of form values** so mid-stream edits don't leak into the in-flight
call. The host re-derives `canGenerate` locally as a defense-in-depth
guard for `handleGenerate` (the panel derives its own copy to drive
its button).

Exported via `lib/components/prompts/index.ts`.

## Fix 1: outside-click-to-close bug — `composedPath()`

Mobile repro: tapping the "+ From dataset" button inside
`ExamplesPanel` would close the side panel. Root cause: Svelte 5
flushes `$state` changes **synchronously** after a click handler
returns, BEFORE the event continues to bubble. So clicking the
button set `pickerOpen = true`, the empty-state `{#if}` block (which
contains the button) was removed, and only THEN did the click bubble
to `<svelte:window>` — where `panelRef.contains(target)` returned
`false` (the target's `compareDocumentPosition` reported
`DISCONNECTED`) and the panel closed. The dataset panel wasn't
affected because no button inside it immediately detaches its own
containing `{#if}` block on click.

Fix: `e.composedPath().includes(panelRef)` instead of
`panelRef.contains(target)`. `composedPath()` is captured at dispatch
time and unaffected by synchronous re-renders. The FAB toggle keeps
`e.stopPropagation()` so the opening click doesn't immediately
re-close the panel (the FAB is outside `panelRef`, so the same
listener would otherwise satisfy the "outside click" condition).

`patterns.md` got a new section documenting the footgun ("Svelte 5
State Flushes Synchronously Inside Event Handlers") so this doesn't
get rediscovered.

## Fix 2: long reasoning line pushed the panel

In a flex row, `flex-1` is `flex: 1 1 0%`, but the **min-width** is
`auto` (content-based) by default. Unbreakable content (a long
reasoning chunk with no whitespace, a URL the model emitted, etc.)
refused to shrink below its intrinsic width and **pushed the side
panel rightward**, off the allotted space. The dataset route's
`DropUploadZone` already has `min-w-0` on its `flex-1` wrapper, which
is why this didn't bite there.

Two changes:

1. **`PromptGenerator.svelte`** — `min-w-0` on the preview wrapper,
   with a comment pointing at the equivalent in
   `routes/datasets/[name]/DropUploadZone.svelte`. Long content is
   now clipped by `GenerationPreview`'s existing `overflow-hidden`
   instead of pushing the panel.
2. **`ReasoningCard.svelte`** — `break-words` on the expanded
   `<pre>`. `whitespace-pre-wrap` already wrapped on whitespace, but
   not on unbreakable sequences; `break-words` (`overflow-wrap:
   break-word`) makes the content actually readable when the preview
   has been shrunk.

**Files touched:**
- New: `yadc/webui/src/lib/components/ui/SidePanel.svelte`
- Renamed: `yadc/webui/src/routes/datasets/[name]/SidePanel.svelte` →
  `yadc/webui/src/routes/datasets/[name]/DatasetSidePanel.svelte`
- New: `yadc/webui/src/lib/components/prompts/PromptSidePanel.svelte`
- Modified: `yadc/webui/src/lib/components/prompts/PromptGenerator.svelte`
  (side-panel layout, `min-w-0` on preview wrapper, snapshot pass to
  `startGeneration`)
- Modified: `yadc/webui/src/lib/components/prompts/index.ts`
  (`PromptSidePanel` export)
- Modified: `yadc/webui/src/lib/components/prompts/ReasoningCard.svelte`
  (`break-words` on expanded `<pre>`)
- Memory docs: `docs/frontend/components-ui.md` (new `SidePanel.svelte`
  entry, with `composedPath()` rationale), `docs/frontend/patterns.md`
  (new "Svelte 5 State Flushes Synchronously Inside Event Handlers"
  section), `docs/frontend/components-domain.md` (new
  `PromptSidePanel.svelte` entry + updated `PromptGenerator.svelte`
  description), `docs/frontend/routes.md` (both routes updated).
