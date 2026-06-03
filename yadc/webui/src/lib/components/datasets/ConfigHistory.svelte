<script lang="ts">
    import {
        fetchConfigHistory,
        restoreConfigHistory,
        type ConfigHistoryEntry
    } from '$lib/stores/configs';
    import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
    import Card from '$lib/components/ui/Card.svelte';
    import ActionBar from '$lib/components/ui/ActionBar.svelte';
    import ActionBarItem from '$lib/components/ui/ActionBarItem.svelte';
    import SvgHistory from '$lib/icons/SvgHistory.svelte';
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgVisibility from '$lib/icons/SvgVisibility.svelte';
    import Alert from '$lib/components/ui/Alert.svelte';
    import { confirmDialog } from '$lib/stores/confirm';

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
    let restoringId: number | null = $state(null);
    let hasMore = $state(false);

    /** Track which entries show full snapshot instead of diff (key = entry id). */
    let showFullMap = $state<Record<number, boolean>>({});

    const PAGE_SIZE = 5;

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

    async function loadHistory(append = false) {
        loading = true;
        error = null;
        try {
            const beforeId =
                append && entries.length > 0 ? entries[entries.length - 1].id : undefined;
            const newEntries = await fetchConfigHistory(datasetName, {
                limit: PAGE_SIZE,
                before_id: beforeId
            });
            hasMore = newEntries.length === PAGE_SIZE;
            if (append) {
                entries = [...entries, ...newEntries];
            } else {
                entries = newEntries;
                showFullMap = {};
            }
        } catch (e) {
            error = e instanceof Error ? e.message : 'Failed to load history';
        } finally {
            loading = false;
        }
    }

    async function handleRestore(entry: ConfigHistoryEntry) {
        if (restoringId !== null) {
            return;
        }
        if (
            !(await confirmDialog.danger(
                'Restore this revision? The current config will be saved to history first.'
            ))
        ) {
            return;
        }
        restoringId = entry.id;
        try {
            await restoreConfigHistory(datasetName, entry.id);
            onsaved?.();
            await loadHistory();
        } catch (e) {
            error = e instanceof Error ? e.message : 'Failed to restore';
        } finally {
            restoringId = null;
        }
    }

    function toggleView(entryId: number) {
        showFullMap = { ...showFullMap, [entryId]: !showFullMap[entryId] };
    }

    $effect(() => {
        void datasetName;
        void configVersion;
        loadHistory();
    });
</script>

<div class="space-y-3 pb-4">
    <!-- Header -->
    <div class="flex items-center justify-between">
        <h3 class="section-heading">Revision History</h3>
        <button
            class="cursor-pointer rounded p-1 text-gray-400 transition-colors hover:bg-gray-700 hover:text-white"
            onclick={() => loadHistory()}
            title="Refresh"
            disabled={loading}
        >
            <SvgRefresh class="h-4 w-4 {loading ? 'animate-spin' : ''}" />
        </button>
    </div>

    {#if error}
        <Alert variant="error" class="text-sm" dismissable ondismiss={() => (error = null)}>
            {error}
        </Alert>
    {/if}

    {#if entries.length === 0 && !loading}
        <div class="py-8 text-center text-sm text-gray-500">
            <p>No revision history yet.</p>
            <p class="mt-1 text-xs text-gray-600">
                History is saved automatically when you make changes.
            </p>
        </div>
    {:else}
        <div class="space-y-3">
            {#each entries as entry, i (entry.id)}
                {@const isLast = i === entries.length - 1}
                {@const showFull = isLast || showFullMap[entry.id]}
                {@const prevEntry = isLast ? null : entries[i + 1]}
                <Card>
                    <div class="flex items-baseline justify-between gap-2 px-3 pt-2 pb-1">
                        <div class="text-xs font-medium text-gray-300">
                            {relativeTime(entry.created_t)}
                        </div>
                        <div class="text-[11px] text-gray-500">
                            {new Date(entry.created_t * 1000).toLocaleString()}
                        </div>
                    </div>
                    <div>
                        {#key showFull}
                            {#if showFull}
                                <TomlEditor
                                    class="text-sm"
                                    value={entry.content}
                                    autoHeight
                                    editable={false}
                                />
                            {:else if prevEntry}
                                <TomlEditor
                                    class="text-sm"
                                    value={entry.content}
                                    original={prevEntry.content}
                                    editable={false}
                                    autoHeight
                                    compactDiff
                                />
                            {/if}
                        {/key}
                    </div>
                    <ActionBar>
                        {#if !isLast}
                            <ActionBarItem
                                onclick={() => toggleView(entry.id)}
                                icon={SvgVisibility}
                                variant="secondary"
                            >
                                {showFull ? 'Show diff' : 'Show full'}
                            </ActionBarItem>
                            <ActionBarItem
                                onclick={() => handleRestore(entry)}
                                disabled={restoringId === entry.id}
                                icon={SvgHistory}
                                variant="primary"
                            >
                                {restoringId === entry.id ? 'Restoring…' : 'Restore'}
                            </ActionBarItem>
                        {:else}
                            <ActionBarItem
                                onclick={() => handleRestore(entry)}
                                disabled={restoringId === entry.id}
                                icon={SvgHistory}
                                variant="primary"
                            >
                                {restoringId === entry.id ? 'Restoring…' : 'Restore'}
                            </ActionBarItem>
                        {/if}
                    </ActionBar>
                </Card>
            {/each}
        </div>

        {#if hasMore}
            <button
                class="btn-secondary w-full text-xs"
                onclick={() => loadHistory(true)}
                disabled={loading}
            >
                {loading ? 'Loading…' : `Show ${PAGE_SIZE} older`}
            </button>
        {/if}
    {/if}
</div>
