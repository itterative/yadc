<script lang="ts">
    import {
        fetchConfigHistory,
        restoreConfigHistory,
        type ConfigHistoryEntry
    } from '$lib/stores/configs';
    import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgFile from '$lib/icons/SvgFile.svelte';

    interface Props {
        datasetName: string;
        /** Incremented by the parent whenever the config is saved externally. */
        configVersion?: number;
        onsaved?: () => void;
    }

    let { datasetName, configVersion = 0, onsaved }: Props = $props();

    let entries: ConfigHistoryEntry[] = $state([]);
    let loading = $state(false);
    let error: string | null = $state(null);
    let selectedId: number | null = $state(null);
    let restoring = $state(false);

    const selected = $derived(entries.find((e) => e.id === selectedId) ?? null);

    // Relative time formatting
    function relativeTime(epochSeconds: number): string {
        const now = Date.now() / 1000;
        const diff = now - epochSeconds;
        if (diff < 60) {
            return 'just now';
        }
        if (diff < 3600) {
            return `${Math.floor(diff / 60)}m ago`;
        }
        if (diff < 86400) {
            return `${Math.floor(diff / 3600)}h ago`;
        }
        return `${Math.floor(diff / 86400)}d ago`;
    }

    function tooltipTime(epochSeconds: number): string {
        return new Date(epochSeconds * 1000).toLocaleString();
    }

    async function loadHistory() {
        loading = true;
        error = null;
        try {
            entries = await fetchConfigHistory(datasetName);
        } catch (e) {
            error = e instanceof Error ? e.message : 'Failed to load history';
        } finally {
            loading = false;
        }
    }

    async function handleRestore(entry: ConfigHistoryEntry) {
        if (restoring) {
            return;
        }
        restoring = true;
        try {
            await restoreConfigHistory(datasetName, entry.id);
            selectedId = null;
            onsaved?.();
            await loadHistory();
        } catch (e) {
            error = e instanceof Error ? e.message : 'Failed to restore';
        } finally {
            restoring = false;
        }
    }

    $effect(() => {
        // Re-fetch when datasetName or external config version changes
        void datasetName;
        void configVersion;
        loadHistory();
    });
</script>

<div class="flex h-full flex-col">
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 class="text-sm font-medium text-gray-300">Revision History</h2>
        <button
            class="cursor-pointer rounded p-1 text-gray-400 transition-colors hover:bg-gray-700 hover:text-white"
            onclick={loadHistory}
            title="Refresh"
            disabled={loading}
        >
            <SvgRefresh class="h-4 w-4 {loading ? 'animate-spin' : ''}" />
        </button>
    </div>

    {#if error}
        <div class="mx-4 mt-3 rounded-md bg-red-900/30 px-3 py-2 text-sm text-red-300">
            {error}
        </div>
    {/if}

    <!-- Content -->
    {#if entries.length === 0 && !loading}
        <div class="flex flex-1 items-center justify-center p-8 text-center text-sm text-gray-500">
            <div>
                <SvgFile class="mx-auto mb-2 h-8 w-8 text-gray-600" />
                <p>No revision history yet.</p>
                <p class="mt-1 text-xs text-gray-600">
                    History is saved automatically when you make changes.
                </p>
            </div>
        </div>
    {:else}
        <div class="flex min-h-0 flex-1">
            <!-- Entry list -->
            <div class="w-40 flex-shrink-0 overflow-y-auto border-r border-border">
                {#if loading && entries.length === 0}
                    <div class="flex items-center justify-center p-4">
                        <SvgRefresh class="h-4 w-4 animate-spin text-gray-500" />
                    </div>
                {/if}
                {#each entries as entry (entry.id)}
                    <button
                        class="flex w-full cursor-pointer flex-col px-3 py-2.5 text-left transition-colors
                            {selectedId === entry.id
                            ? 'bg-gray-700/60 text-white'
                            : 'text-gray-300 hover:bg-gray-800/60'}"
                        onclick={() => (selectedId = entry.id)}
                    >
                        <span class="text-xs font-medium" title={tooltipTime(entry.created_t)}>
                            {relativeTime(entry.created_t)}
                        </span>
                        <span class="mt-0.5 text-[11px] text-gray-500">
                            {new Date(entry.created_t * 1000).toLocaleString()}
                        </span>
                    </button>
                {/each}
            </div>

            <!-- Preview pane -->
            <div class="flex min-w-0 flex-1 flex-col">
                {#if selected}
                    <div class="flex items-center justify-between border-b border-border px-3 py-2">
                        <span class="text-xs text-gray-400">
                            Snapshot from {new Date(selected.created_t * 1000).toLocaleString()}
                        </span>
                        <button
                            class="flex cursor-pointer items-center gap-1.5 rounded bg-accent px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                            onclick={() => handleRestore(selected)}
                            disabled={restoring}
                        >
                            <SvgRefresh class="h-3 w-3" />
                            {restoring ? 'Restoring…' : 'Restore'}
                        </button>
                    </div>
                    <div class="min-h-0 flex-1 overflow-y-auto">
                        <TomlEditor value={selected.content} editable={false} class="h-full" />
                    </div>
                {:else}
                    <div class="flex flex-1 items-center justify-center text-sm text-gray-500">
                        Select a revision to preview
                    </div>
                {/if}
            </div>
        </div>
    {/if}
</div>
