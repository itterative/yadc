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
