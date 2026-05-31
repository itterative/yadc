<script lang="ts">
    import { createTabsState, setTabsContext } from './TabsContext.svelte';

    interface Props {
        class?: string;
        value?: string;
        hideSingle?: boolean;
        children: import('svelte').Snippet;
    }

    let {
        class: className = '',
        value = $bindable(''),
        hideSingle = false,
        children
    }: Props = $props();

    const state = createTabsState();
    setTabsContext(state);

    let currentValue = $derived(state.tabs[state.activeIndex]?.id ?? '');

    $effect(() => {
        value = currentValue;
    });

    $effect(() => {
        const idx = state.tabs.findIndex((t) => t.id === value);
        if (idx !== -1 && idx !== state.activeIndex) {
            state.setActiveIndex(idx);
        }
    });
</script>

<div class="flex flex-col {className}">
    <!-- Segmented control header -->
    {#if !(hideSingle && state.tabs.length <= 1)}
        <div class="flex items-center gap-1 rounded-lg bg-gray-800/50 p-1" role="tablist">
            {#each state.tabs as t, i (t.id)}
                {@const isActive = state.activeIndex === i}
                <button
                    type="button"
                    role="tab"
                    aria-selected={isActive}
                    class="flex flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors {isActive
                        ? 'bg-surface text-white shadow-sm'
                        : 'text-gray-400 hover:text-gray-200'}"
                    onclick={() => state.setActiveIndex(i)}
                >
                    {#if t.icon}
                        {@const Icon = t.icon}
                        <Icon class="h-3.5 w-3.5" />
                    {/if}
                    {t.label}
                </button>
            {/each}
        </div>
    {/if}

    <!-- Tab content -->
    <div class="mt-4" role="tabpanel">
        {@render children()}
    </div>
</div>
