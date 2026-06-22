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

**Files touched:**
- `yadc/webui/src/lib/components/prompts/PromptSidePanel.svelte`
- `yadc/webui/src/lib/components/prompts/GenerationPreview.svelte`
- `docs/frontend/components-domain.md` (`PromptSidePanel` +
  `GenerationPreview` entries)
