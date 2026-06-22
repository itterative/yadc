---
date: 2026-06-22
---
# FAB = cancel during streaming + `FabButton` primitive

**Context:** On mobile, the `ui/SidePanel` FAB sat on top of the
streaming cancel button in `GenerationPreview`'s footer (FAB at
`fixed right-6 bottom-6 z-20` vs. the footer with no z-index), so
the user couldn't cancel without first closing whatever was covering
it. The dataset side panel didn't have this problem because its
captioning cancel lives inside the `Caption` tab's `ActionBar`. The
prompts side panel has no equivalent in-panel cancel (the form is
snapshotted into the in-flight call, so opening the panel mid-stream
is pointless — the only useful mid-stream action is **cancel**).
Goal: make the FAB context-sensitive on the prompts page — toggle
when idle, **cancel** while streaming — and lift the FAB's
styling/positioning into a reusable primitive so the snippet stays
trivial.

**User decision:** the prompts page only — the dataset side panel
stays alone for now (could opt in later via the same `fab` snippet).

## `FabButton.svelte` primitive

New `yadc/webui/src/lib/components/ui/FabButton.svelte` — the FAB
shape used by `SidePanel` (and available to future callers). Props:
`icon: Component<{class?:string}>`, `label: string`, `onclick: () => void`,
`variant?: 'accent' | 'error'` (default `accent`). The component
owns the `h-16 w-16 rounded-full text-white shadow-lg
transition-colors` shape, the `h-6 w-6` icon size, and the
accent/error color mapping (`bg-accent hover:bg-accent-hover` vs.
`bg-error hover:bg-error/80`). `title` and `aria-label` both come
from `label` (the FAB is icon-only, so `label` is the only text AT
announces). Icon is passed as a `Component` — same pattern as
`ActionBarItem`.

## `SidePanel.svelte` — `primaryAction` → `fab` + wrapper-level `stopPropagation`

Two changes beyond the original `primaryAction` sketch:

1. **Renamed `primaryAction` → `fab`.** Short, matches the visual
   metaphor (it's the FAB), and pairs naturally with the
   `FabButton` it usually renders.
2. **Moved `e.stopPropagation()` from the snippet's `onclick` to
   the wrapper's `onclick`.** The positioning wrapper is the
   common ancestor of the FAB and `<svelte:window>`, so it can
   short-circuit the bubble centrally. Callers no longer need to
   remember to call `stopPropagation` in their snippet — they
   provide a plain `onclick` and the wrapper guarantees the click
   never reaches the outside-click listener. The default toggle's
   `stopPropagation` was load-bearing (the opening click would
   otherwise re-close the panel via the bubble-phase listener) and
   is now structural.

The wrapper carries `<!-- svelte-ignore a11y_click_events_have_key_events -->`
+ `<!-- svelte-ignore a11y_no_static_element_interactions -->` —
the div's `onclick` is purely a propagation guard, not an
interaction; the actual control is the `<button>` inside `FabButton`,
so the a11y rules for interactive elements don't apply.

The default FAB is now a one-liner: `<FabButton icon={SvgMenuLeft}
label="Toggle panel" onclick={() => (open = !open)} />`.

## `PromptSidePanel.svelte` — `fab` snippet

`{#snippet fab()}` with two branches:

- `isStreaming` → `<FabButton icon={SvgClose} variant="error"
  label="Cancel generation" onclick={cancelGenerationState} />`
- otherwise → `<FabButton icon={SvgMenuLeft} label="Toggle panel"
  onclick={() => (open = !open)} />`

The `isStreaming` derivation stays local to the panel
(`$derived(generation.status === 'streaming')`), so the host doesn't
need to know about the FAB's state machine.

## `GenerationPreview.svelte` — footer cancel is desktop-only

The footer cancel is now `class="hidden lg:flex ..."`. On mobile the
FAB is the only cancel; on desktop the FAB is hidden (`lg:hidden`
in the wrapper) and the footer cancel is the only cancel. No
overlap, no dead DOM behind the FAB. The `cancelGenerationState`
import stays — the desktop path still needs it.

## Edge case (not fixed in this iteration)

On mobile, if the user opens the side panel mid-stream, the FAB is
hidden behind the panel (`z-20` < panel `z-40`) and the footer
cancel is hidden (`hidden`). So there's no visible cancel until
the user closes the panel — at which point the FAB appears. Two
taps (close panel → tap FAB to cancel). Acceptable for now; future
options include an in-panel cancel button (mirroring the dataset
pattern) or a panel-aware FAB.

**Files touched:**
- New: `yadc/webui/src/lib/components/ui/FabButton.svelte`
- Modified: `yadc/webui/src/lib/components/ui/SidePanel.svelte`
  (renamed prop, wrapper-level `stopPropagation`, default uses
  `FabButton`)
- Modified: `yadc/webui/src/lib/components/prompts/PromptSidePanel.svelte`
  (`fab` snippet with streaming-cancel / default-toggle branches)
- Modified: `yadc/webui/src/lib/components/prompts/GenerationPreview.svelte`
  (footer cancel is `hidden lg:flex`)
- Memory docs: `docs/frontend/components-ui.md` (new
  `FabButton.svelte` entry; updated `SidePanel.svelte` entry
  mentions `FabButton` and the `fab` snippet), `docs/frontend/components-domain.md`
  (updated `PromptSidePanel.svelte` entry).
