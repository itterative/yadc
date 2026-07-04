<script lang="ts">
    /**
     * A curate-by-toggle list for one of the always-add / banned policies
     * (the new Tags-section counterpart to :comp:`TierList`'s visual
     * highlight tiers). Reads the global ``tagHighlights`` store for
     * quick-add affordances, and a sibling list (passed in via the
     * ``list`` getter) for the active policy. The ``list`` getter keeps
     * it agnostic over which of the two policies it represents: the
     * parent (``TagSettings``) passes a derivation that resolves to
     * ``$tagPolicy.alwaysAdd`` or ``$tagPolicy.banned``.
     *
     * Layout:
     *
     * - **Quick add** (top): every entry from the relevant tier that's
     *   NOT already in the policy renders as a small "+ tag" button.
     *   Clicking adds it to the policy in one step — the user never has
     *   to type a tag they already curated in Customize.
     * - **Active policy** (below): each entry renders as a removable
     *   chip with the same × pattern as :comp:`TierList`. Hovering
     *   adds the inline "Clear all" button when more than one entry
     *   is present (matches TierList's affordance surface).
     * - **Custom additions** (bottom): the existing :comp:`TagInput`
     *   primitive for typing tags not in either list.
     */

    import { type Readable } from 'svelte/store';
    import { tagHighlights, type TagTier, type TagChipView } from '$lib/stores/tagging';
    import TagChip from './TagChip.svelte';
    import TagInput from './TagInput.svelte';

    interface Props {
        /** Section heading (e.g., "Always add"). */
        title: string;
        /** Short description below the heading. */
        description: string;
        /** Reactive view of the active policy list (alwaysAdd / banned). */
        list: Readable<string[]>;
        /** Tier to source quick-add entries from (starred / undesired). */
        tier: TagTier;
        /** Add a tag to the policy. Called with the trimmed name. */
        onadd: (tag: string) => void;
        /** Remove a tag from the policy. */
        onremove: (tag: string) => void;
        /** Fill classes for chips in the active policy row. Tier-coloured
         *  so the user can tell at a glance which policy list they're in. */
        activeFillClass: string;
    }

    let { title, description, list, tier, onadd, onremove, activeFillClass }: Props =
        $props();

    /* The chip model for both rows. Keeping a single shape (TagChipView)
     * lets us share :comp:`TagChip` for both; the visible difference
     * between the two rows is purely ``isCustom`` (controls the ×) and
     * the fill classes the parent passes down. */

    /** Tier members NOT yet in the policy → render as "+ add" buttons. */
    let quickAddChips = $derived.by<TagChipView[]>(() => {
        const tierList = tier === 'starred' ? $tagHighlights.starred : $tagHighlights.undesired;
        const present = new Set($list);
        return tierList
            .filter((t) => !present.has(t))
            .map((t) => ({ tag: t, score: 1.0, isCustom: false }));
    });

    /** Active policy → render as filled, removable chips. */
    let activeChips = $derived.by<TagChipView[]>(() => {
        const list_ = $list;
        const tierSet = new Set(
            tier === 'starred' ? $tagHighlights.starred : $tagHighlights.undesired
        );
        return list_.map<TagChipView>((t) => ({
            tag: t,
            score: -1,
            isCustom: !tierSet.has(t) // × only on custom (non-tier) chips; tier members are managed in Customize
        }));
    });

    /* ``TagInput`` dedupes against this map (tag → "category"); the value
     * isn't meaningful here — just needs a per-tag bucket so a duplicate
     * add is rejected. */
    let existingCustomTags = $derived(new Map($list.map((t) => [t, title])));
</script>

<section>
    <div class="mb-3 flex flex-col gap-1 pb-1">
        <h4 class="section-heading font-semibold text-gray-400 uppercase">{title}</h4>
        <p class="text-xs text-gray-600">{description}</p>
    </div>

    {#if quickAddChips.length > 0}
        <div class="mb-2 flex flex-wrap gap-1.5">
            {#each quickAddChips as chip (chip.tag)}
                <button
                    type="button"
                    class="cursor-pointer rounded-md border border-dashed border-gray-600 px-2 py-1 text-xs text-gray-400 transition-colors hover:border-gray-400 hover:text-gray-200"
                    title="Add {chip.tag} to {title}"
                    onclick={() => onadd(chip.tag)}
                >
                    {chip.tag}
                </button>
            {/each}
        </div>
    {/if}

    <div class="flex flex-wrap items-center gap-1.5">
        {#each activeChips as chip (chip.tag)}
            <TagChip
                {chip}
                enabled={true}
                starred={tier === 'starred' && chip.isCustom === false}
                fillClass={activeFillClass}
                removable={chip.isCustom}
                onremovecustom={onremove}
                ontoggle={(tag) => !chip.isCustom && onremove(tag)}
            />
        {/each}
        <TagInput category={title} modelTags={[]} {existingCustomTags} {onadd} />
    </div>
</section>
