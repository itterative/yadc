<script lang="ts">
    import type { Snippet } from 'svelte';
    import Tabs from './Tabs.svelte';

    interface Props {
        class?: string;
        value?: string;
        storageId?: string;
        hideSingle?: boolean;
        end?: Snippet;
        children: Snippet;
    }

    let {
        class: className = '',
        value = $bindable(''),
        storageId,
        hideSingle = false,
        end,
        children
    }: Props = $props();
</script>

<Tabs class={className} bind:value {storageId} {hideSingle} {end}>
    {#snippet tab(t, isActive, onclick)}
        <button
            type="button"
            role="tab"
            aria-selected={isActive}
            class="cursor-pointer rounded-full px-4 py-1.5 text-sm font-medium transition-colors {isActive
                ? 'bg-accent text-white'
                : 'text-gray-400 hover:bg-white/5 hover:text-gray-200'}"
            {onclick}
        >
            {t.label}
        </button>
    {/snippet}
    {@render children()}
</Tabs>
