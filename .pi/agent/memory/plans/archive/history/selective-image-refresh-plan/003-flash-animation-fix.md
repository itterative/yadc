---
date: 2026-05-27
---

# Phase 2.1: Flash animation on tile update + infinite-loop fix

**Context:** After Phase 2, testing revealed two issues: (1) `$effect` blocks reading `images` then writing to `images` caused `effect_update_depth_exceeded` infinite loops, breaking the UI entirely; (2) there was no visual indication of which tiles had just been updated during captioning.

**Decision:** Wrapped `images.map()` calls in `untrack()` so only the store subscriptions drive the effects. Added a `flash` timestamp field to `ImageInfo` — set to `Date.now()` when per-image events arrive. `DatasetImage.svelte` watches `item.flash` with `$effect`, toggles a `flashing` class that plays a `tile-flash` CSS animation (accent border glow fading over 0.8s), and auto-clears on `animationend`.

**Rationale:** `untrack` is the standard Svelte 5 pattern for writing to a reactive signal from an effect without subscribing to it. The flash animation uses `outline` (same property as `.selected`) so there's no layout shift, and `animationend` avoids needing timers.

**Files touched:** `yadc/webui/src/lib/stores/datasetImages.ts`, `yadc/webui/src/lib/components/dataset/DatasetImage.svelte`, `yadc/webui/src/routes/datasets/[name]/+page.svelte`
