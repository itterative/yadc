<script lang="ts">
    import type { Snippet } from 'svelte';
    import { createTabsState, setTabsContext } from './TabsContext.svelte';
    import type { TabItem } from './TabsContext.svelte';
    import { loadStoredTabId, saveStoredTabId } from './tabState';
    import TabScroller from './TabScroller.svelte';

    interface Props {
        class?: string;
        value?: string;
        /** When set, the active tab id is persisted to localStorage under
         *  ``yadc/tab:<storageId>`` and restored on mount, so a refresh returns
         *  the user to the same tab. Must be unique across tab instances. */
        storageId?: string;
        hideSingle?: boolean;
        end?: Snippet;
        tab?: Snippet<[TabItem, boolean, () => void]>;
        children: Snippet;
    }

    let {
        class: className = '',
        value = $bindable(''),
        storageId,
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

    // Persist the active tab across reloads when ``storageId`` is given.
    // Restore writes ``activeIndex`` directly rather than going through ``value``:
    // the sync effects above run in declaration order, so routing through
    // ``value`` would let the ``active -> value`` effect clobber it before the
    // pull effect runs. One-shot restore so later user clicks win.
    let restored = false;
    $effect(() => {
        if (restored || !storageId || state.tabs.length === 0) {
            return;
        }
        restored = true;
        const stored = loadStoredTabId(storageId);
        if (stored) {
            const idx = state.tabs.findIndex((t) => t.id === stored);
            if (idx !== -1) {
                state.setActiveIndex(idx);
            }
        }
    });
    $effect(() => {
        const cur = currentValue;
        if (!storageId || !restored || !cur) {
            return;
        }
        saveStoredTabId(storageId, cur);
    });
</script>

<div class="flex flex-col {className}">
    {#if !(hideSingle && state.tabs.length <= 1)}
        <div class="relative flex items-center border-b border-border">
            <TabScroller class="flex-1 px-2 py-2">
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
            </TabScroller>
            {#if end}
                <div class="flex shrink-0 items-center py-2 pr-2">
                    {@render end()}
                </div>
            {/if}
        </div>
    {/if}

    <div class="flex min-h-0 flex-1 flex-col" role="tabpanel">
        {@render children()}
    </div>
</div>
