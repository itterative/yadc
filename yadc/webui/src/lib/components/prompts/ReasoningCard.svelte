<script lang="ts">
    import { tick } from 'svelte';
    import SvgChevronDown from '$lib/icons/SvgChevronDown.svelte';
    import SvgChevronUp from '$lib/icons/SvgChevronUp.svelte';
    import { generation } from '$lib/stores/prompts';

    let expanded = $state(false);
    let scrollEl: HTMLPreElement | undefined = $state();

    // The card only appears once the model has produced any reasoning.
    let visible = $derived(generation.reasoning.length > 0);

    // Last non-empty line of the reasoning — used as the collapsed summary.
    let lastLine = $derived.by(() => {
        const r = generation.reasoning;
        if (!r) {
            return '';
        }
        const lines = r.split('\n');
        for (let i = lines.length - 1; i >= 0; i--) {
            if (lines[i].length > 0) {
                return lines[i];
            }
        }
        return r;
    });

    // Jump to the bottom when the card is opened (or re-opened) so the
    // user lands on the freshest content. Deferred to the next tick so
    // the freshly-rendered <pre> has its layout computed.
    $effect(() => {
        if (expanded) {
            tick().then(() => {
                if (scrollEl) {
                    scrollEl.scrollTop = scrollEl.scrollHeight;
                }
            });
        }
    });

    // Follow the stream while the user is near the bottom. If they've
    // scrolled up to read, leave them alone.
    $effect(() => {
        // Track the reasoning so this re-runs on every chunk.
        const _reasoning = generation.reasoning;
        void _reasoning;
        if (!expanded || !scrollEl) {
            return;
        }
        const distFromBottom = scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight;
        if (distFromBottom < 50) {
            scrollEl.scrollTop = scrollEl.scrollHeight;
        }
    });
</script>

{#if visible}
    <div class="rounded-lg bg-gray-800 flex flex-col overflow-hidden mb-4">
        <button
            type="button"
            class="flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-left text-sm text-gray-400"
            onclick={() => (expanded = !expanded)}
            aria-expanded={expanded}
        >
            {#if expanded}
                <SvgChevronUp class="h-3.5 w-3.5 shrink-0" />
            {:else}
                <SvgChevronDown class="h-3.5 w-3.5 shrink-0" />
            {/if}
            <span class="shrink-0 font-medium text-gray-300">Reasoning:</span>
            <span class="min-w-0 flex-1 truncate" class:text-gray-500={expanded}>
                {lastLine || '…'}
            </span>
        </button>
        {#if expanded}
            <pre
                bind:this={scrollEl}
                class="max-h-48 overflow-y-auto px-3 py-2 font-mono text-sm whitespace-pre-wrap text-gray-400">{generation.reasoning}</pre>
        {/if}
    </div>
{/if}
