<script lang="ts">
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgStar from '$lib/icons/SvgStar.svelte';
    import {
        displayTag,
        tagSettings,
        type TagChipView,
        type CategoryOption
    } from '$lib/stores/tagging';
    import SvgChevronDown from '$lib/icons/SvgChevronDown.svelte';
    import { anchorVisible, positionPopover } from './popover';

    interface Props {
        chip: TagChipView;
        enabled: boolean;
        starred: boolean;
        /** Fill classes (border + bg + text) for the chip body. Computed by
         *  the parent from the tier + enabled state so this leaf stays free
         *  of the tier colour rules. */
        fillClass: string;
        ontoggle?: (tag: string) => void;
        onremovecustom?: (tag: string) => void;
        /** Whether this chip shows a remove (×) button. The parent decides —
         *          the chip itself has no policy: custom non-starred tags are
         *          removable in the per-image grid, every tier member is removable
         *          in the customize panel, and starred tags in the grid are
         *          toggle-only (curated elsewhere). */
        removable?: boolean;
        /** When provided (and the chip is starred), renders a caret that opens
         *  a popover to reassign the tag's section. Writes to a global override
         *  store via the parent. */
        onchangecategory?: (tag: string, category: string | null) => void;
        /** Category options for the reassignment popover. */
        categoryOptions?: readonly CategoryOption[];
        /** Current override value (``null`` = Auto / not overridden), so the
         *  popover can mark the active option. */
        currentCategory?: string | null;
    }

    let {
        chip,
        enabled,
        starred,
        fillClass,
        ontoggle,
        onremovecustom,
        onchangecategory,
        categoryOptions,
        currentCategory = null,
        removable = false
    }: Props = $props();

    let menuEl = $state<HTMLDivElement | null>(null);
    let caretEl = $state<HTMLButtonElement | null>(null);

    /** Visible label — ``chip.tag`` is the identity (canonical for curated
     *  chips, display-form for prune-grid model tags); project it through the
     *  user's ``replaceUnderscores`` preference for rendering. Callbacks
     *  still receive the raw ``chip.tag`` identity. */
    let label = $derived(
        displayTag(
            { name: chip.tag, canonical_form: chip.canonicalForm },
            $tagSettings.replaceUnderscores
        )
    );

    /** Display label for the badge — the short form, looked up from the
     *  options so it stays in sync with the popover's active highlight. */
    let categoryShort = $derived(
        categoryOptions?.find((o) => o.value === (currentCategory ?? null))?.short ?? 'Auto'
    );

    let showCategoryChange = $derived(
        starred && onchangecategory !== undefined && categoryOptions !== undefined
    );

    function toggleMenu(e: MouseEvent) {
        e.stopPropagation();
        if (menuEl?.matches(':popover-open')) {
            menuEl.hidePopover();
        } else if (menuEl && caretEl) {
            positionPopover(menuEl, caretEl);
        }
    }

    function pickCategory(e: MouseEvent, value: string | null) {
        e.stopPropagation();
        onchangecategory?.(chip.tag, value);
        menuEl?.hidePopover();
    }

    // Hide the popover if its anchor scrolled away — reposition otherwise.
    function onScroll() {
        if (menuEl?.matches(':popover-open') && caretEl) {
            if (!anchorVisible(caretEl)) {
                menuEl.hidePopover();
            } else {
                positionPopover(menuEl, caretEl);
            }
        }
    }
</script>

<!--
    ``chip.tag`` is the tag identity, whose form depends on context:
    canonical (``speech_bubble``) for curated chips in the Customize tab,
    display-form (``speech bubble``) for model-output chips in the prune
    grid. The visible label projects it through the user's
    ``replaceUnderscores`` preference (see ``label``); callbacks
    (``ontoggle`` / ``onremovecustom`` / ``onchangecategory``) receive
    the raw ``chip.tag`` identity so each consumer matches in its own
    convention.
-->

<div
    role="button"
    tabindex="0"
    aria-pressed={enabled}
    aria-label={!chip.canonicalForm
        ? `${label} (custom tag, click to toggle)`
        : chip.absent
          ? `${label} (starred, not detected in this image, click to toggle)`
          : `${label}, confidence ${Math.round(chip.score * 100)}%`}
    class="flex cursor-pointer justify-center gap-1 rounded-md border px-2 py-1 text-xs transition-colors {fillClass}"
    onclick={() => ontoggle?.(chip.tag)}
    onkeydown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            ontoggle?.(chip.tag);
        }
    }}
    title={!chip.canonicalForm
        ? 'Custom tag'
        : chip.absent
          ? 'Starred — not detected in this image'
          : chip.score.toFixed(4)}
>
    {#if starred}
        <SvgStar class="h-3 w-3 self-center text-amber-400" />
    {/if}

    <span class="mr-1.5">
        {label}

        {#if chip.score >= 0 && chip.canonicalForm && !chip.absent}
            <span class="text-[95%] opacity-70">
                {Math.round(chip.score * 100)}%
            </span>
        {/if}
    </span>

    {#if showCategoryChange}
        <button
            bind:this={caretEl}
            type="button"
            class="inline-flex cursor-pointer items-center gap-0.5 rounded bg-gray-600/40 px-1 opacity-80 transition-colors hover:opacity-100 focus:opacity-100"
            title="Change section"
            aria-label="Change section for {label}"
            onclick={toggleMenu}
        >
            <span class="text-[90%] tracking-wide uppercase">{categoryShort}</span>
            <SvgChevronDown class="ml-1 h-3 w-3" />
        </button>
    {/if}

    {#if removable}
        <button
            type="button"
            class="inline-flex flex-0 cursor-pointer items-center justify-center rounded px-1 opacity-70 transition-colors hover:bg-purple-700/30 hover:opacity-100 focus:opacity-100"
            title="Remove custom tag"
            aria-label="Remove {label}"
            onclick={(e) => {
                e.stopPropagation();
                onremovecustom?.(chip.tag);
            }}
        >
            <SvgClose class="h-3 w-3" />
        </button>
    {/if}
</div>

{#if showCategoryChange}
    <div
        bind:this={menuEl}
        popover="auto"
        style="position: fixed; margin: 0;"
        class="overflow-hidden rounded-md border border-border bg-surface py-0.5 shadow-xl"
        role="menu"
        onbeforetoggle={(e) => {
            if (e.newState === 'open' && menuEl && caretEl) {
                positionPopover(menuEl, caretEl);
            }
        }}
    >
        {#each categoryOptions as opt (String(opt.value))}
            <button
                type="button"
                role="menuitem"
                class="block w-full cursor-pointer px-3 py-1 text-left text-xs transition-colors hover:bg-gray-700 {currentCategory ===
                opt.value
                    ? 'text-accent'
                    : 'text-gray-200'}"
                onclick={(e) => pickCategory(e, opt.value)}
            >
                {opt.label}
            </button>
        {/each}
    </div>
{/if}

<svelte:window onscroll={onScroll} />
