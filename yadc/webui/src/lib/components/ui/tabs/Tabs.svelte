<script lang="ts">
    import type { Snippet } from 'svelte';
    import { createTabsState, setTabsContext } from './TabsContext.svelte';
    import type { TabItem } from './TabsContext.svelte';

    interface Props {
        class?: string;
        value?: string;
        hideSingle?: boolean;
        end?: Snippet;
        tab?: Snippet<[TabItem, boolean, () => void]>;
        children: Snippet;
    }

    let {
        class: className = '',
        value = $bindable(''),
        hideSingle = false,
        end,
        tab,
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
    {#if !(hideSingle && state.tabs.length <= 1)}
        <div class="flex items-center border-b border-border px-2 py-2" role="tablist">
            <div class="flex-1"></div>
            <div class="flex flex-wrap justify-center gap-1">
                {#each state.tabs as t, i (t.id)}
                    {@const isActive = state.activeIndex === i}
                    {#if tab}
                        {@render tab(t, isActive, () => state.setActiveIndex(i))}
                    {:else}
                        <button
                            type="button"
                            role="tab"
                            aria-selected={isActive}
                            class="cursor-pointer"
                            onclick={() => state.setActiveIndex(i)}
                        >
                            {t.label}
                        </button>
                    {/if}
                {/each}
            </div>
            <div class="flex flex-1 justify-end">
                {#if end}
                    {@render end()}
                {/if}
            </div>
        </div>
    {/if}

    <div class="flex min-h-0 flex-1 flex-col" role="tabpanel">
        {@render children()}
    </div>
</div>
