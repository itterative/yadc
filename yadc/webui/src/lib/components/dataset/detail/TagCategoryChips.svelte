<script lang="ts">
    import type { SvelteSet } from 'svelte/reactivity';
    import TagInput from './TagInput.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';

    /** One rendered chip — score + custom-flag computed by the parent so
     *  this component can stay focused on layout. */
    export interface TagChipView {
        tag: string;
        score: number;
        isCustom: boolean;
    }

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
        ontoggle: (tag: string) => void;
        /** Remove a custom tag entirely (drop from enabled AND from the
         *  custom-tags map). Only rendered for custom chips via an inline
         *  × button. */
        onremovecustom: (tag: string) => void;
        /** Submit a new custom tag for this category. The parent owns the
         *  category binding; this callback only emits the tag name. */
        onaddcustom: (tag: string) => void;
        /** Re-enable every model tag in this category (custom are by
         *  definition enabled, so Select all is a no-op for them). */
        onselectall: () => void;
        /** Disable model tags in this category AND drop custom tags
         *  assigned to it — explicit "remove everything in this category". */
        onclear: () => void;
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
        onclear
    }: Props = $props();

    const CATEGORIES_WITHOUT_CUSTOM_INPUT_DEFAULT: ReadonlySet<string> = new Set(['rating']);

    let customInputEnabled = $derived(!CATEGORIES_WITHOUT_CUSTOM_INPUT_DEFAULT.has(category));

    /** Section-level header controls (Select all / Clear) are only shown
     *  when there's at least one chip. The custom-input toggle is
     *  independent of that — it's about *adding*, not pruning. */
    let hasChips = $derived(chips.length > 0);
</script>

<section>
    <div class="mb-3 flex items-center justify-end gap-3 text-xs text-gray-400">
        <h4 class="flex-1 font-semibold text-gray-400 uppercase">{category}</h4>

        {#if hasChips}
            <button class="cursor-pointer hover:text-gray-200" onclick={onselectall}>
                Select all
            </button>
            <span class="text-gray-600">·</span>
            <button class="cursor-pointer hover:text-gray-200" onclick={onclear}> Clear </button>
        {/if}
    </div>

    <div class="flex flex-wrap gap-1.5">
        {#if !customInputEnabled && !hasChips}
            <span class="pl-1 text-xs text-gray-400 italic">(none)</span>
        {/if}

        {#each chips as chip (chip.tag)}
            {@const isEnabled = enabled.has(chip.tag)}
            <div
                role="button"
                tabindex="0"
                aria-pressed={isEnabled}
                aria-label={chip.isCustom
                    ? `${chip.tag} (custom tag, click to toggle)`
                    : `${chip.tag}, confidence ${Math.round(chip.score * 100)}%`}
                class="flex cursor-pointer justify-center rounded-md border px-2 py-1 text-xs transition-colors {!isEnabled
                    ? 'border-gray-700 bg-gray-800 text-gray-500 line-through opacity-60 hover:border-gray-600'
                    : chip.isCustom
                      ? 'border-purple-500/60 bg-purple-500/20 text-purple-300'
                      : 'border-accent/60 bg-accent/20 text-accent'}"
                onclick={() => ontoggle(chip.tag)}
                onkeydown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        ontoggle(chip.tag);
                    }
                }}
                title={chip.isCustom ? 'Custom tag' : chip.score.toFixed(4)}
            >
                <span>{chip.tag}</span>
                {#if !chip.isCustom}
                    <span class="ml-1.5 text-[95%] opacity-70">
                        {Math.round(chip.score * 100)}%
                    </span>
                {:else}
                    <button
                        type="button"
                        class="ml-2 inline-flex flex-0 items-center justify-center rounded px-1 opacity-70 transition-colors hover:bg-purple-700/30 hover:opacity-100 focus:opacity-100"
                        title="Remove custom tag"
                        aria-label={`Remove ${chip.tag}`}
                        onclick={(e) => {
                            e.stopPropagation();
                            onremovecustom(chip.tag);
                        }}
                    >
                        <SvgClose class="h-3 w-3" />
                    </button>
                {/if}
            </div>
        {/each}
        {#if customInputEnabled}
            <TagInput {category} {modelTags} existingCustomTags={customTags} onadd={onaddcustom} />
        {/if}
    </div>
</section>
