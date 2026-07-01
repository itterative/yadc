<script lang="ts">
    import type { SvelteSet } from 'svelte/reactivity';
    import TagInput from './TagInput.svelte';
    import TagChip from './TagChip.svelte';
    import {
        CATEGORIES_WITHOUT_CUSTOM_INPUT,
        TAG_TIERS,
        type TagChipView,
        type TagTier
    } from '$lib/stores/tagging';

    interface Props {
        /** Category name shown in the section header. */
        category: string;
        /** Pre-sorted chips to render (custom first alphabetical, then model
         *  by score descending). */
        chips: readonly TagChipView[];
        /** Model-output tags in this category — passed to :comp:`TagInput`
         *  for dedupe (re-adding a pruned model tag re-enables it). */
        tags: readonly string[];
        /** All currently-added custom tags. :comp:`TagInput` dedupes against
         *  this so the user can't add the same custom tag twice. */
        customTags: ReadonlyMap<string, string>;
        /** Shared enabled set (model + custom). Drives the chip's on/off state. */
        enabled: SvelteSet<string>;
        /** Toggle a chip. Custom tags follow the same toggle semantics as
         *  model tags (click re-enables / disables). Use :prop:`onRemoveCustom`
         *  to drop a custom tag entirely. */
        ontoggle?: (tag: string) => void;
        /** Remove a custom tag entirely (drop from enabled AND from the
         *  custom-tags map). Only rendered for non-starred custom chips —
         *  starred tags are curated in the customize tab, never removed here. */
        onremovecustom: (tag: string) => void;
        /** Submit a new custom tag for this category. The parent owns the
         *  category binding; this callback only emits the tag name. */
        onaddcustom: (tag: string) => void;
        /** Re-enable every model tag in this category (custom are by
         *  definition enabled, so Select all is a no-op for them). */
        onselectall?: () => void;
        /** Clear model tags in this category AND drop custom tags
         *  assigned to it — explicit "remove everything in this category". */
        onclear?: () => void;
        /** Optional highlight-tier map. When a tag resolves to a tier, the
         *          chip gets a colored background so the user can spot curated
         *          tags at a glance. Pure visual — it does not change toggle /
         *          save semantics. */
        tiers?: ReadonlyMap<string, TagTier>;
    }

    let {
        category,
        chips,
        tags: modelTags,
        customTags,
        enabled,
        ontoggle,
        onremovecustom,
        onaddcustom,
        onselectall,
        onclear,
        tiers
    }: Props = $props();

    let customInputEnabled = $derived(!CATEGORIES_WITHOUT_CUSTOM_INPUT.has(category));

    /** Section-level header controls (Select all / Clear) are only shown
     *  when there's at least one chip. The custom-input toggle is
     *  independent of that — it's about *adding*, not pruning. */
    let hasChips = $derived(chips.length > 0);

    /** Fill classes (border + bg + text) for a highlighted tier. When the
     *  chip is enabled the tier overrides the default custom / model fill so
     *  curated tags stand out by background, not just an outline. Disabled
     *  tier chips keep the muted gray fill but pick up a faint tier-tinted
     *  border so the category is still recognizable when toggled off. */
    function tierFillClass(tag: string, enabled: boolean): string {
        switch (tiers?.get(tag)) {
            case TAG_TIERS.starred:
                return enabled
                    ? 'border-amber-400/60 bg-amber-400/20 text-amber-300'
                    : 'border-amber-400/25';
            case TAG_TIERS.desired:
                return enabled
                    ? 'border-emerald-400/60 bg-emerald-400/20 text-emerald-300'
                    : 'border-emerald-400/25';
            case TAG_TIERS.undesired:
                return enabled
                    ? 'border-rose-400/60 bg-rose-400/20 text-rose-300'
                    : 'border-rose-400/25';
            default:
                return '';
        }
    }

    function isStarred(tag: string): boolean {
        return tiers?.get(tag) === TAG_TIERS.starred;
    }
</script>

<section>
    <div class="mb-3 flex items-center justify-end gap-3 text-xs text-gray-400">
        <h4 class="flex-1 font-semibold text-gray-400 uppercase">{category}</h4>

        {#if hasChips}
            {#if onselectall}
                <button class="cursor-pointer hover:text-gray-200" onclick={onselectall}>
                    Select all
                </button>
            {/if}

            {#if onselectall && onclear}
                <span class="text-gray-600">·</span>
            {/if}

            {#if onclear}
                <button class="cursor-pointer hover:text-gray-200" onclick={onclear}>
                    Clear
                </button>
            {/if}
        {/if}
    </div>

    <div class="flex flex-wrap gap-1.5">
        {#if !customInputEnabled && !hasChips}
            <span class="pl-1 text-xs text-gray-400 italic">(none)</span>
        {/if}

        {#each chips as chip (chip.tag)}
            {@const isEnabled = enabled.has(chip.tag)}
            {@const starred = isStarred(chip.tag)}
            {@const tierFill = tierFillClass(chip.tag, isEnabled)}
            {@const baseFill = tierFill
                ? tierFill
                : isEnabled
                  ? chip.isCustom
                      ? 'border-purple-500/60 bg-purple-500/20 text-purple-300'
                      : 'border-accent/60 bg-accent/20 text-accent'
                  : ''}
            {@const disabledFill =
                !isEnabled && !tierFill
                    ? 'border-gray-700 bg-gray-800 text-gray-500 line-through opacity-60 hover:border-gray-600'
                    : !isEnabled
                      ? 'bg-gray-800 text-gray-500 line-through opacity-60 hover:brightness-125'
                      : 'hover:brightness-110'}
            <TagChip
                {chip}
                enabled={isEnabled}
                {starred}
                fillClass={`${disabledFill} ${baseFill}`}
                {ontoggle}
                {onremovecustom}
                removable={chip.isCustom && !starred}
            />
        {/each}
        {#if customInputEnabled}
            <TagInput {category} {modelTags} existingCustomTags={customTags} onadd={onaddcustom} />
        {/if}
    </div>
</section>
