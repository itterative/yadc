<script lang="ts">
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import SvgHistory from '$lib/icons/SvgHistory.svelte';
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import { confirmDialog } from '$lib/stores/confirm';
    import { toast } from '$lib/stores/toasts';
    import { createAbortContext } from '$lib/abort';
    import {
        deleteHistoryEntry,
        fetchHistoryEntry,
        fetchHistoryList
    } from '$lib/stores/prompts/api';
    import type { PromptHistoryEntry, PromptHistoryListItem } from '$lib/stores/prompts';
    import { friendlyErrorMessage } from '$lib/api';
    import { formatRelativeTime } from '$lib/format';

    interface Props {
        /** Called when the user picks an entry to restore. The host
         *  is responsible for switching to the right tab + populating
         *  the form state from the entry. */
        onrestore: (entry: PromptHistoryEntry) => void;
    }

    let { onrestore }: Props = $props();

    const abort = createAbortContext();

    let entries = $state<PromptHistoryListItem[]>([]);
    let loaded = $state(false);
    let isLoading = $state(false);
    let isRestoringId = $state<number | null>(null);
    let error: string | null = $state(null);

    $effect(() => {
        return () => abort.abort();
    });

    export async function refresh(): Promise<void> {
        isLoading = true;
        error = null;
        try {
            entries = await fetchHistoryList(abort.signal);
            loaded = true;
        } catch (e) {
            if (abort.signal.aborted) {
                return;
            }
            error = friendlyErrorMessage(e, 'Failed to load history');
        } finally {
            if (!abort.signal.aborted) {
                isLoading = false;
            }
        }
    }

    $effect(() => {
        refresh();
    });

    async function handleRestore(item: PromptHistoryListItem) {
        if (isRestoringId !== null) {
            return;
        }
        isRestoringId = item.id;
        try {
            const entry = await fetchHistoryEntry(item.id, abort.signal);
            onrestore(entry);
        } catch (e) {
            if (abort.signal.aborted) {
                return;
            }
            toast.error(friendlyErrorMessage(e, 'Failed to restore entry'));
        } finally {
            if (!abort.signal.aborted) {
                isRestoringId = null;
            }
        }
    }

    async function handleDelete(item: PromptHistoryListItem, event: MouseEvent) {
        event.stopPropagation();
        const ok = await confirmDialog.danger({
            title: 'Delete history entry',
            message: 'This entry will be permanently removed.',
            confirmLabel: 'Delete'
        });
        if (!ok) {
            return;
        }
        try {
            await deleteHistoryEntry(item.id, abort.signal);
            entries = entries.filter((e) => e.id !== item.id);
            toast.success('History entry deleted');
        } catch (e) {
            if (abort.signal.aborted) {
                return;
            }
            toast.error(friendlyErrorMessage(e, 'Failed to delete entry'));
        }
    }
</script>

<div class="flex h-full min-h-0 flex-col">
    <div class="flex items-center justify-between border-b border-border px-4 py-2">
        <div class="flex items-center gap-2 text-sm font-medium text-gray-300">
            <SvgHistory class="h-4 w-4" />
            Saved prompts
        </div>
        <button
            class="flex cursor-pointer items-center gap-1.5 rounded-md p-1.5 text-gray-400 transition-colors hover:bg-bg hover:text-white disabled:opacity-50"
            onclick={() => refresh()}
            disabled={isLoading}
            title="Refresh"
        >
            {#if isLoading}
                <SvgSpinner class="h-4 w-4 animate-spin" />
            {:else}
                <SvgRefresh class="h-4 w-4" />
            {/if}
        </button>
    </div>

    <div class="min-h-0 flex-1 overflow-auto p-3">
        {#if !loaded && isLoading}
            <SpinnerBlock class="py-12" size="h-4 w-4" label="Loading history…" />
        {:else if error}
            <div class="alert-error text-sm">{error}</div>
        {:else if entries.length === 0}
            <div
                class="flex h-full flex-col items-center justify-center gap-2 px-4 py-12 text-center"
            >
                <SvgHistory class="h-8 w-8 text-muted" />
                <p class="text-sm text-gray-400">No saved prompts yet</p>
                <p class="text-xs text-gray-500">
                    Click "Save to history" in the Generate or Refine tab to add one.
                </p>
            </div>
        {:else}
            <p class="mb-3 px-1 text-center text-xs text-amber-400 italic">
                Restoring replaces your current form values.
            </p>
            <ul class="space-y-2">
                {#each entries as item (item.id)}
                    <li>
                        <!-- Display-only card — actions live in the footer. The whole
                             entry used to be a ``role="button"`` that restored on click;
                             on mobile that was both undiscoverable (no hover to reveal
                             the "Restore" label) and easy to fire by accident. -->
                        <div
                            class="group block w-full rounded-lg border border-border bg-bg/40 p-3 text-left transition-colors hover:border-gray-500 hover:bg-bg/60"
                            data-testid="history-entry"
                        >
                            <!-- Intent preview -->
                            <p
                                class="line-clamp-2 text-sm leading-snug text-gray-200 group-hover:text-white"
                            >
                                {item.intent_preview}
                            </p>

                            <!-- Badges row -->
                            <div class="mt-2 flex flex-wrap items-center gap-1.5">
                                <span
                                    class="rounded px-1.5 py-0.5 text-xs font-medium {item.mode ===
                                    'refine'
                                        ? 'bg-purple-500/20 text-purple-300'
                                        : 'bg-accent/20 text-accent'}"
                                >
                                    {item.mode}
                                </span>
                                <span
                                    class="rounded bg-gray-700/60 px-1.5 py-0.5 text-xs text-gray-300"
                                >
                                    {item.focus}
                                </span>
                                {#if item.example_count > 0}
                                    <span
                                        class="rounded bg-gray-700/60 px-1.5 py-0.5 text-xs text-gray-300"
                                    >
                                        {item.example_count} example{item.example_count === 1
                                            ? ''
                                            : 's'}
                                    </span>
                                {/if}
                                {#if item.had_template}
                                    <span
                                        class="rounded bg-blue-500/20 px-1.5 py-0.5 text-xs text-blue-300"
                                        title="This entry was a refine with a template"
                                    >
                                        with template
                                    </span>
                                {/if}
                            </div>

                            <!-- Footer: timestamp + actions -->
                            <div class="mt-2 flex items-center justify-between">
                                <span
                                    class="text-xs text-gray-500"
                                    title={new Date(item.created_t * 1000).toLocaleString()}
                                >
                                    {formatRelativeTime(item.created_t)}
                                </span>
                                <span class="flex items-center gap-2">
                                    <!-- Restore — always visible (SvgHistory reused;
                                         semantic match for "restore from history"). -->
                                    <button
                                        type="button"
                                        class="rounded p-1 text-gray-400 cursor-pointer transition-colors hover:bg-bg hover:text-accent focus:text-accent focus:outline-none disabled:cursor-wait disabled:opacity-50"
                                        onclick={() => handleRestore(item)}
                                        disabled={isRestoringId !== null}
                                        title="Restore"
                                        aria-label="Restore"
                                        data-testid="history-restore"
                                    >
                                        {#if isRestoringId === item.id}
                                            <SvgSpinner class="h-4.5 w-4.5 animate-spin" />
                                        {:else}
                                            <SvgHistory class="h-4.5 w-4.5" />
                                        {/if}
                                    </button>
                                    <!-- Delete — always visible, same discoverability
                                         reasoning as restore (mobile has no hover). -->
                                    <button
                                        type="button"
                                        class="rounded p-1 text-gray-400 cursor-pointer transition-colors hover:bg-bg hover:text-error focus:text-error focus:outline-none"
                                        onclick={(e) => handleDelete(item, e)}
                                        title="Delete entry"
                                        data-testid="history-delete"
                                    >
                                        <SvgDelete class="h-4.5 w-4.5" />
                                    </button>
                                </span>
                            </div>
                        </div>
                    </li>
                {/each}
            </ul>
        {/if}
    </div>
</div>
