---
date: 2026-07-01
---
# Starred / desired / undesired highlight tiers

`tagHighlights` storable (`yadc/tagHighlights`, $version 1) holds
three mutually-exclusive tier lists plus a `categoryOverrides` map
(starred tag → forced section). `tagTierMap` is derived
(tag → tier lookup) for the per-image prune grid; visual fill is
applied in place so the user can spot curated tags at a glance.

Tiers are **purely visual** here — starred tags are pulled to the
lead of their destination section in the prune grid and force-shown
even when the model didn't detect them; desired/undesired chips are
just green/red tinted. Tier membership doesn't trigger any backend
behaviour; it's curation data the user owns.

The phase also split out `TagCategoryChips` (per-image grid section)
and `TagInput` (inline custom-tag input) from `dataset/detail/`
into `lib/components/tagging/`, then built the curate-side
counterparts: `TagCustomize` (the new tab in the batch side panel),
`TierList` (one tier's curation chip row with `×` to remove +
per-chip category-reassignment popover on starred), and a small
`SvgStar` icon for the Starred column.

**Files (backend):** `yadc/api/controllers/api_tagging.py`
(`TagImageBody` comments-only refresh, no behavioural change here),
`yadc/api/services/tagging.py` (cache-key documentation now mentions
the visual tier data indirectly via `tagTierMap`; no service-layer
change for the tiers themselves), `yadc/taggers/__init__.py`,
`yadc/taggers/base.py` (docstring on `TaggerResult.customizations`
clarifies it stays independent of the tier system).

**Files (frontend):** `yadc/webui/src/lib/stores/tagging/highlights.ts`
new storable + accessors, `yadc/webui/src/lib/stores/tagging/constants.ts`
new `TAG_CATEGORIES` / `STARRED_CATEGORY_OPTIONS` shared with both
the prune grid and the curate popover,
`yadc/webui/src/lib/stores/tagging/selectionModel.ts` new pure
selection-model logic (`computeOrderedCategories` for grid layout +
`computeSelection` for the merged pruned-result / customizations
snapshot one-pass builder; `starredSection` is the single source of
truth for where a starred tag files so display + save +
persistence can't drift).

**Components:** `yadc/webui/src/lib/components/tagging/{TagCustomize,
TierList,TagCategoryChips,TagChip,TagInput,popover}.svelte`,
`yadc/webui/src/lib/components/tagging/TagSettings.svelte`
(the old `TagSettingsPanel` body split out),
`yadc/webui/src/lib/components/tagging/TagSettingsPanel.svelte`
(thinned down to a CompactPillTabs host: Settings / Customize + the
sticky Start/Stop footer),
`yadc/webui/src/lib/components/dataset/detail/TagCategoryChips.svelte`
removed (replaced by the tagging/ copy),
`yadc/webui/src/lib/components/dataset/detail/Tags.svelte` trimmed
to use the new shared widgets,
`yadc/webui/src/lib/components/ui/tabs/CompactPillTabs.svelte`
minor hover-state tweak.

**Docs:** `tagger-architecture.md` WebUI surface section now
describes the curated tiers; `frontend/components-domain.md`
adds `TierList` / `TagCustomize` / `TagCategoryChips` entries;
`frontend/stores.md` adds `highlights.ts` / `selectionModel.ts`.
