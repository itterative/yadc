---
date: 2026-06-22
---
# Footer: drop status text, move cancel in from preview

UX refinement.

**`PromptSidePanel.svelte`** — footer: dropped the contextual
status `<p>` on the left (info was redundant with the tabs
themselves and crowded the footer on narrow widths). The
Generate/Refine primary CTA now swaps to a `btn-danger` Cancel
mid-stream, so the cancel is reachable from inside the panel on
both layouts — closing the gap from 015 where, on mobile with the
panel open during streaming, neither the FAB nor the preview
footer cancel was visible.

**`GenerationPreview.svelte`** — cancel button removed (now in the
panel footer). `cancelGenerationState` import dropped. Header's
`SvgClose` "Clear preview" stays (different action — `reset`).

The FAB cancel and the footer cancel both call
`cancelGenerationState`, so behaviour is the same regardless of
which the user reaches for.

**Files touched (so far):**
- `yadc/webui/src/lib/components/prompts/PromptSidePanel.svelte`
- `yadc/webui/src/lib/components/prompts/GenerationPreview.svelte`
- `docs/frontend/components-domain.md` (`PromptSidePanel` +
  `GenerationPreview` entries)

## History: per-entry restore icon (no more click-anywhere)

The whole entry was a `role="button"` that triggered restore on
click. On mobile that's both undiscoverable (no hover to reveal
the "Restore" label) and easy to fire by accident. Replaced with
an always-visible `SvgHistory` icon button (the section header
already uses `SvgHistory` — semantic match for "restore from
history"). The card is no longer a button: `role`/`tabindex`/
`onclick`/`onkeydown` removed, `cursor-pointer` and `focus:*`
dropped, no `disabled:*` on the entry itself. Restore shows a
spinner in place of the icon when this entry is restoring.
Delete moved out of its hover-only wrapper to always-visible too
— same discoverability reasoning (mobile has no hover, so the
delete was unreachable there).

**Files touched:**
- `yadc/webui/src/lib/components/prompts/PromptHistoryPanel.svelte`
- `docs/frontend/components-domain.md` (`PromptHistoryPanel`
  entry)

## Generate/Refine closes the panel

PromptSidePanel's Generate/Refine primary CTA now closes the
panel before calling `ongenerate`. Initially scoped to refine,
expanded to both modes (no good reason to leave the panel open
when the preview starts streaming). Done by wrapping the
callback in a local `handleGenerateClick` that sets
`open = false` first.

**Files touched:**
- `yadc/webui/src/lib/components/prompts/PromptSidePanel.svelte`

## GenerationPreview: inline status below the stream

The "Generating…" footer row moved inline, right under the `<pre>`,
where it actually describes what's being watched. Other terminal
states got the same treatment.

- `<pre>` wrapped in `{#if generation.body}` with a comment
  explaining why ("a bit ugly, but necessary to keep the
  whitespace of the template") — without the wrapper, an empty
  `<pre>` rendered awkwardly in some status transitions. Earlier
  tried `<p>` (no wrapper needed) but reverted to `<pre>` for
  semantic / a11y reasons (screen readers announce `<pre>` as
  preformatted text and respect whitespace).
- Caret uses `class:hidden={generation.status !== 'streaming'}`
  instead of `{#if}` — same result, more idiomatic.
- Inline status row (`mt-2 flex justify-center gap-1.5 text-sm`):
  - `streaming && !generation.body` → spinner + "Generating…"
    (only before the first token — once text streams in, the
    caret is the in-progress signal and a redundant label would
    compete with it).
  - `cancelled` → close + "Cancelled" (`text-warning` —
    user-initiated, so warning instead of error red).
  - **No "Done" indicator** — body being complete + Copy/Save
    buttons appearing in the header is enough.
- Error state unchanged — its full empty-state UI already
  displays text.
- Footer now variables-only; hidden entirely when
  `variables.length === 0`.

**Files touched:**
- `yadc/webui/src/lib/components/prompts/GenerationPreview.svelte`

## Streaming auto-scroll action

The "follow the stream + jump to bottom on open" logic that
lived in `ReasoningCard.svelte` (two `$effect` blocks + a
`scrollEl` ref + a `tick()` defer) is now a reusable Svelte
action in `lib/actions/autoscroll.ts` — same pattern as the
existing `autosize.ts`. Applied to both the reasoning card
`<pre>` and `GenerationPreview`'s body `<pre>` so the streamed
template follows the same auto-follow behaviour.

API:
```ts
use:autoscroll                              // defaults: scrollOnMount true
use:autoscroll={{ scrollOnMount: false }}   // don't jump on mount
```

Internals: a `MutationObserver` with `childList` + `characterData`
+ `subtree` options, so it catches both new child nodes and
text-content updates (which is what Svelte does when re-rendering
`{text}` bindings). On mount, a `requestAnimationFrame` callback
sets `scrollTop = scrollHeight` if `scrollOnMount` is true (defer
to next frame so the freshly-rendered DOM has its layout
computed).

Pause/resume: the original design used a position threshold
(`distFromBottom < 50` → auto-scroll), which meant small scroll
gestures within the threshold didn't disable it — users had to
scroll "aggressively" up to escape auto-follow. Replaced with
gesture-based detection: a `scroll` event listener compares
`scrollTop` against the previous value. Since our own
programmatic scroll only ever sets `scrollTop = scrollHeight`
(max position), any *decrease* in `scrollTop` between events is
unambiguously user-initiated (mouse wheel, touch drag, or
keyboard) and pauses auto-scroll. Resume happens only when the
user scrolls back to within 5px of the bottom.

Where to attach it: an earlier attempt put `use:autoscroll` on
`GenerationPreview`'s `<pre>`, but that didn't work because the
`<pre>` itself wasn't the scroll container (the body `<div>`
was, via `overflow-auto`) — `scrollTop` on the `<pre>` was a
no-op. Moved the action to the parent `<div>`.

The MutationObserver is disconnected in `destroy()`, so the
action is safe to attach to conditionally-rendered elements
(ReasoningCard's `<pre>` only exists when the card is expanded).

**Files touched:**
- `yadc/webui/src/lib/actions/autoscroll.ts` (new)
- `yadc/webui/src/lib/components/prompts/ReasoningCard.svelte`
  (dropped two `$effect`s + `scrollEl` ref + `tick` import;
  added `use:autoscroll`)
- `yadc/webui/src/lib/components/prompts/GenerationPreview.svelte`
  (added `use:autoscroll` on the body scroll container, not
  the `<pre>`)
- `.pi/agent/memory/frontend-architecture.md` (`lib/actions/`
  folder added to the current-layout one-liner, alongside
  `autosize.ts`)
- `.pi/agent/memory/docs/frontend/components-domain.md`
  (`ReasoningCard` + `GenerationPreview` entries mention
  `use:autoscroll`)
