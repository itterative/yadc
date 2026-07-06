<script lang="ts">
    /** A single tier's curation list for the customize panel. Unlike
     *  :comp:`TagCategoryChips` (a per-image grid section with Select-all /
     *  Clear and model-tag concepts), this is the management surface for one
     *  highlight tier: every chip is removable (the × drops the tag from the
     *  tier), and the starred tier additionally offers per-chip section
     *  reassignment. Reads the global highlights store directly so the parent
     *  (:comp:`TagCustomize`) stays a thin layout + description shell. */

    import { SvelteSet } from 'svelte/reactivity';
    import {
        displayTag,
        tagHighlights,
        tagCategoryOverrideMap,
        setTagTier,
        removeTagTier,
        setTagCategoryOverride,
        removeTagCategoryOverride,
        tagSettings,
        TAG_TIERS,
        STARRED_CATEGORY_OPTIONS,
        type TagTier
    } from '$lib/stores/tagging';
    import TagChip from './TagChip.svelte';
    import TagInput from './TagInput.svelte';

    interface Props {
        tier: TagTier;
        label: string;
        description: string;
        /** Header label on chips' category badge when the tier is starred
         *          (the section-reassignment popover is only shown then). */
        allowCategoryOverride: boolean;
    }

    let { tier, description, label, allowCategoryOverride }: Props = $props();

    // Chips to render: tier members as custom chips so they carry a remove ×.
    // Membership is binary here (no "disabled but kept" state), so the
    // enabled set mirrors the tier exactly — clicking the chip body drops
    // the tag too.
    let chips = $derived(
        tier === TAG_TIERS.starred
            ? $tagHighlights.starred
            : tier === TAG_TIERS.desired
              ? $tagHighlights.desired
              : $tagHighlights.undesired
    );

    // Canonical name strings for iteration / lookup (entries are stored
    // canonical; display is projected at render by :comp:`TagChip`).
    // Canonical name strings for iteration / lookup (entries are stored
    // canonical; display is projected at render by :comp:`TagChip`).
    let chipNames = $derived(chips.map((e) => e.name));

    let enabled = $derived.by(() => {
        const s = new SvelteSet<string>();
        for (const name of chipNames) {
            s.add(name);
        }
        return s;
    });

    // TagInput dedupes against this map (tag → "category"); the values are
    // only used for dedupe, so the tier value stands in for a category.
    let customTags = $derived(new Map(chipNames.map((name) => [name, tier])));

    // Tier fill is computed by TagCategoryChips' tierFillClass normally;
    // here each chip belongs to ``tier`` so the fill is constant per list.
    let tierFill = $derived(
        tier === TAG_TIERS.starred
            ? 'border-amber-400/60 bg-amber-400/20 text-amber-300'
            : tier === TAG_TIERS.desired
              ? 'border-emerald-400/60 bg-emerald-400/20 text-emerald-300'
              : 'border-rose-400/60 bg-rose-400/20 text-rose-300'
    );

    function handleChangeCategory(tag: string, category: string | null) {
        if (category === null) {
            removeTagCategoryOverride(tag);
        } else {
            setTagCategoryOverride(tag, category);
        }
    }
</script>

<section>
    <div class="mb-3 flex flex-col gap-1 pb-1 text-xs text-gray-400">
        <h4 class="flex-1 font-semibold text-gray-400 uppercase">{label}</h4>
        <p class="text-xs text-gray-600">{description}</p>
    </div>

    <div class="flex flex-wrap gap-1.5">
        {#each chips as entry (entry.name)}
            <TagChip
                chip={{ tag: entry.name, score: 1.0, canonicalForm: entry.canonical_form }}
                enabled={enabled.has(entry.name)}
                starred={tier === TAG_TIERS.starred}
                fillClass={tierFill}
                onremovecustom={removeTagTier}
                removable={true}
                onchangecategory={allowCategoryOverride ? handleChangeCategory : undefined}
                categoryOptions={allowCategoryOverride ? STARRED_CATEGORY_OPTIONS : undefined}
                currentCategory={$tagCategoryOverrideMap.get(
                    displayTag(entry, $tagSettings.replaceUnderscores)
                ) ?? null}
            />
        {/each}
        <TagInput
            category={label}
            modelTags={[]}
            existingCustomTags={customTags}
            onadd={(entry) => setTagTier(tier, entry)}
        />
    </div>
</section>
