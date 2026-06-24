<script lang="ts">
    import type { Snippet } from 'svelte';
    import { onMount } from 'svelte';

    interface Props {
        class?: string;
        children: Snippet;
    }

    let { class: className = '', children }: Props = $props();

    // Track which edges still have content to scroll to; only that side's
    // fade is applied so tabs at the extremes aren't dimmed when everything
    // fits. The fade is a CSS mask on the scrolling row (rather than a
    // colored overlay), so tabs dissolve into whatever background is behind
    // them — surface, a translucent track, anything — without color matching.
    let el: HTMLElement | undefined = $state();
    let content: HTMLElement | undefined = $state();
    let atStart = $state(true);
    let atEnd = $state(true);

    function update() {
        if (!el) {
            return;
        }
        atStart = el.scrollLeft <= 1;
        // 1px tolerance for sub-pixel rounding.
        atEnd = Math.ceil(el.scrollLeft) + el.clientWidth >= el.scrollWidth - 1;
    }

    // Compose the mask from up to two gradient layers (left and/or right).
    // ``mask-composite: intersect`` keeps the intersection so each edge fades
    // independently; an empty string clears it (no dimming) when nothing
    // overflows.
    let mask = $derived.by(() => {
        const fade = '28px';
        const layers: string[] = [];
        if (!atStart) {
            layers.push(`linear-gradient(to right, transparent, black ${fade})`);
        }
        if (!atEnd) {
            layers.push(`linear-gradient(to left, transparent, black ${fade})`);
        }
        if (layers.length === 0) {
            return '';
        }
        const value = layers.join(', ');
        return `-webkit-mask-image: ${value}; mask-image: ${value}; -webkit-mask-composite: source-in; mask-composite: intersect;`;
    });

    onMount(() => {
        update();
        const ro = new ResizeObserver(update);
        if (el) {
            ro.observe(el);
        }
        if (content) {
            ro.observe(content);
        }
        return () => ro.disconnect();
    });
</script>

<div class="min-w-0 {className}">
    <div bind:this={el} class="tab-scroller flex overflow-x-auto" onscroll={update} style={mask}>
        <!-- ``w-max`` sizes the row to its content so it overflows (and thus
             scrolls) when the tabs are wider than the viewport; ``mx-auto``
             re-centers it when it fits, preserving the centered look on
             wide screens. -->
        <div
            bind:this={content}
            class="mx-auto flex items-center justify-center gap-1 md:w-max md:min-w-full"
            role="tablist"
        >
            {@render children()}
        </div>
    </div>
    <!-- Accent edge-glow on top of the masked content — a faint hint that
         there's more to scroll toward, even when a tab is fully off-screen.
         Only shown on the side(s) with content beyond the viewport. -->
    <div
        class="pointer-events-none absolute inset-y-0 left-0 w-7 transition-opacity duration-150 {atStart
            ? 'opacity-0'
            : 'opacity-100'}"
        style="background: linear-gradient(to right, color-mix(in srgb, var(--color-accent) 18%, transparent), transparent)"
    ></div>
    <div
        class="pointer-events-none absolute inset-y-0 right-0 w-7 transition-opacity duration-150 {atEnd
            ? 'opacity-0'
            : 'opacity-100'}"
        style="background: linear-gradient(to left, color-mix(in srgb, var(--color-accent) 18%, transparent), transparent)"
    ></div>
</div>

<style>
    /* Hide the scrollbar — the edge mask signals that more tabs exist. */
    .tab-scroller {
        scrollbar-width: none;
    }
    .tab-scroller::-webkit-scrollbar {
        display: none;
    }
</style>
