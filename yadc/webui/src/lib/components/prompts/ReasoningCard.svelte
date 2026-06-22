<script lang="ts">
    import SvgChevronDown from '$lib/icons/SvgChevronDown.svelte';
    import SvgChevronUp from '$lib/icons/SvgChevronUp.svelte';
    import { generation } from '$lib/stores/prompts';
    import { autoscroll } from '$lib/actions/autoscroll';

    let expanded = $state(false);

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
</script>

{#if visible}
    <div class="mb-4 flex flex-col overflow-hidden rounded-lg bg-gray-800">
        <button
            type="button"
            class="flex w-full cursor-pointer items-center gap-2 p-4 text-left text-sm text-gray-400"
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
                use:autoscroll
                class="max-h-48 overflow-y-auto px-3 py-2 font-mono text-sm wrap-break-word whitespace-pre-wrap text-gray-400">{generation.reasoning}</pre>
        {/if}
    </div>
{/if}
