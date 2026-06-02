<script lang="ts">
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgCheck from '$lib/icons/SvgCheck.svelte';
    import SvgCopy from '$lib/icons/SvgCopy.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgSparkle from '$lib/icons/SvgSparkle.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { confirmDialog } from '$lib/stores/confirm';
    import { captionOptions } from '$lib/stores/captionActions';
    import type { CaptionData, HistoryEntry, ImageInfo } from '$lib/stores/datasetImages';
    import { friendlyErrorMessage } from '$lib/api';
    import { PasswordPromptCancelled } from '$lib/stores/passwordPrompt';

    interface Props {
        item: ImageInfo;
        captionData: CaptionData | null;
        isLoadingCaption: boolean;
        captionError: string | null;
        isCaptioning: boolean;
        historyEntries: HistoryEntry[];
        isLoadingHistory: boolean;
        isRestoring: boolean;
        isHistoryExpanded: boolean;
        copiedKey: string | null;
        onCaptioningStart: () => Promise<void>;
        onCaptioningCancel: () => Promise<void>;
        onSaveCaption: (text: string) => Promise<void>;
        onRestoreHistory: (index: number) => Promise<void>;
        onHistoryToggle: () => void;
        onCopy: (key: string, text: string) => void;
    }

    let {
        item,
        captionData,
        isLoadingCaption,
        captionError,
        isCaptioning,
        historyEntries,
        isLoadingHistory,
        isRestoring,
        isHistoryExpanded,
        copiedKey,
        onCaptioningStart,
        onCaptioningCancel,
        onSaveCaption,
        onRestoreHistory,
        onHistoryToggle,
        onCopy
    }: Props = $props();

    let isEditing = $state(false);
    let editCaption = $state('');
    let isSavingCaption = $state(false);
    let isCancelling = $state(false);
    let captioningError: string | null = $state(null);

    let activeDraftName = $derived($captionOptions?.draft?.trim() || '');

    // Reset local state when item changes
    $effect(() => {
        // Track item.id as a dependency
        void item.id;
        isEditing = false;
        captioningError = null;
    });

    // Sync editCaption from captionData when not editing
    $effect(() => {
        if (!isEditing) {
            editCaption = captionData?.caption || '';
        }
    });

    function handleStartEdit() {
        editCaption = captionData?.caption || '';
        isEditing = true;
    }

    function handleCancelEdit() {
        isEditing = false;
        editCaption = captionData?.caption || '';
    }

    async function handleSave() {
        if (!isEditing) {
            return;
        }
        isSavingCaption = true;
        try {
            await onSaveCaption(editCaption);
            isEditing = false;
        } catch {
            // Parent already set captionError — stay in edit mode so the user can retry
        } finally {
            isSavingCaption = false;
        }
    }

    async function handleCaptionImage() {
        captioningError = null;
        try {
            await onCaptioningStart();
        } catch (e) {
            if (e instanceof PasswordPromptCancelled) {
                return;
            }
            captioningError = friendlyErrorMessage(e, 'Failed to caption image');
        }
    }

    async function handleCancelSingleCaptioning() {
        isCancelling = true;
        try {
            await onCaptioningCancel();
        } finally {
            isCancelling = false;
        }
    }

    async function handleRestoreClick(index: number) {
        const ok = await confirmDialog.warning(
            'Restore this revision? The current caption and extras will be saved to history first.'
        );
        if (!ok) {
            return;
        }
        try {
            await onRestoreHistory(index);
        } catch {
            // Parent already set captionError
        }
    }

    function autosize(el: HTMLTextAreaElement) {
        function resize() {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 360) + 'px';
        }
        resize();
        return { update: resize };
    }
</script>

<div class="space-y-4">
    <!-- Caption -->
    <div>
        <h3 class="mb-2 text-sm font-medium text-gray-300">Caption</h3>

        {#if isLoadingCaption}
            <SpinnerBlock size="h-4 w-4" label="Loading caption..." />
        {:else if captionError}
            <p class="text-sm text-error">{captionError}</p>
        {:else}
            <div class="relative overflow-hidden rounded-lg bg-gray-800">
                <!-- Copy button — anchored to the caption box, not the scrollable
                     text area, so it stays put while the user scrolls the text. -->
                {#if captionData && captionData.caption && !isEditing && !isCaptioning}
                    <button
                        type="button"
                        class="absolute top-2 right-2 z-10 cursor-pointer rounded p-1 text-gray-400 transition-colors hover:bg-gray-700 hover:text-gray-200"
                        aria-label="Copy caption"
                        title="Copy caption"
                        onclick={() => onCopy('caption', captionData!.caption!)}
                    >
                        {#if copiedKey === 'caption'}
                            <SvgCheck class="h-4 w-4 text-success" />
                        {:else}
                            <SvgCopy class="h-4 w-4" />
                        {/if}
                    </button>
                {/if}

                <!-- Text area — only this part scrolls. Overlays are siblings
                     of the scroll container so they stay fixed over the visible
                     area instead of scrolling with the content. -->
                <div class="relative">
                    <div class="max-h-60 overflow-y-auto">
                        {#if isEditing}
                            <textarea
                                bind:value={editCaption}
                                class="w-full resize-none bg-transparent p-3 font-mono text-sm whitespace-pre-wrap text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
                                placeholder="Enter caption..."
                                use:autosize
                                oninput={(e) => {
                                    const el = e.currentTarget;
                                    el.style.height = 'auto';
                                    el.style.height = Math.min(el.scrollHeight, 360) + 'px';
                                }}
                            ></textarea>
                        {:else if captionData}
                            {#if captionData.caption}
                                <pre
                                    class="p-3 pr-10 text-sm whitespace-pre-wrap text-gray-200">{captionData.caption}</pre>
                            {:else}
                                <pre
                                    class="p-3 text-sm whitespace-pre-wrap text-gray-500 italic">No caption</pre>
                            {/if}
                        {/if}
                    </div>

                    {#if isCaptioning}
                        <div
                            class="absolute inset-0 z-10 flex items-center justify-center gap-2 bg-black/60 text-sm text-gray-300"
                        >
                            <SvgSpinner class="h-4 w-4 animate-spin" />
                            <span
                                >{activeDraftName
                                    ? `Generating draft…`
                                    : 'Generating caption…'}</span
                            >
                        </div>
                    {:else if captioningError}
                        <div
                            class="absolute inset-x-0 top-0 z-10 bg-error/90 px-3 py-1.5 text-center text-sm text-white"
                        >
                            {captioningError}
                        </div>
                    {/if}
                </div>

                <!-- Action bar — footer of the caption box. -->
                <div class="grid grid-cols-2 gap-2 bg-black/15 p-2 text-sm">
                    {#if isEditing}
                        <div class="flex items-center justify-center">
                            <button
                                class="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-gray-400 transition-colors hover:bg-gray-700 hover:text-gray-200 disabled:cursor-not-allowed disabled:opacity-50"
                                onclick={handleCancelEdit}
                                disabled={isSavingCaption}
                            >
                                <SvgClose class="h-4 w-4 shrink-0" />
                                <span class="min-w-0 truncate">Cancel</span>
                            </button>
                        </div>
                        <div class="flex items-center justify-center">
                            <button
                                class="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-accent transition-colors hover:bg-gray-700 hover:text-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                                onclick={handleSave}
                                disabled={isSavingCaption}
                            >
                                <SvgCheck class="h-4 w-4 shrink-0" />
                                <span class="min-w-0 truncate"
                                    >{isSavingCaption ? 'Saving…' : 'Save'}</span
                                >
                            </button>
                        </div>
                    {:else if isCaptioning}
                        <div class="col-span-2 flex items-center justify-center">
                            <button
                                class="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-error transition-colors hover:bg-gray-700 hover:text-error/80 disabled:cursor-not-allowed disabled:opacity-50"
                                onclick={handleCancelSingleCaptioning}
                                disabled={isCancelling}
                            >
                                <SvgClose class="h-4 w-4 shrink-0" />
                                <span class="min-w-0 truncate"
                                    >{isCancelling ? 'Cancelling…' : 'Cancel'}</span
                                >
                            </button>
                        </div>
                    {:else if captionData}
                        <div class="flex items-center justify-center">
                            <button
                                class="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-accent transition-colors hover:bg-gray-700 hover:text-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                                onclick={handleCaptionImage}
                            >
                                <SvgSparkle class="h-4 w-4 shrink-0" />
                                <span class="min-w-0 truncate"
                                    >{activeDraftName
                                        ? `Draft (${activeDraftName})`
                                        : 'Caption'}</span
                                >
                            </button>
                        </div>
                        <div class="flex items-center justify-center">
                            <button
                                class="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-gray-400 transition-colors hover:bg-gray-700 hover:text-gray-200"
                                onclick={handleStartEdit}
                            >
                                <SvgEdit class="h-4 w-4 shrink-0" />
                                <span class="min-w-0 truncate">Edit</span>
                            </button>
                        </div>
                    {/if}
                </div>
            </div>
        {/if}
    </div>

    <!-- Drafts -->
    {#if captionData?.drafts && Object.keys(captionData.drafts).length > 0}
        <div>
            <h3 class="mb-2 text-sm font-medium text-gray-300">Drafts</h3>
            <p class="mb-2 text-xs text-gray-500">
                Alternative captions you can reference from the prompt template (e.g.
                <code class="text-gray-400">drafts.&lt;name&gt;</code>) and export later.
            </p>
            <div class="space-y-2">
                {#each Object.entries(captionData.drafts) as [name, text] (name)}
                    {@const draftKey = `draft:${name}`}
                    <div class="relative rounded-lg bg-gray-800 p-3">
                        <p class="mb-1 text-xs font-medium text-accent">{name}</p>
                        <pre
                            class="max-h-40 overflow-y-auto pr-9 font-mono text-sm whitespace-pre-wrap text-gray-200">{text}</pre>
                        {#if text}
                            <button
                                type="button"
                                class="absolute top-2 right-2 cursor-pointer rounded p-1 text-gray-400 transition-colors hover:bg-gray-700 hover:text-gray-200"
                                aria-label={`Copy draft "${name}"`}
                                title={`Copy draft "${name}"`}
                                onclick={() => onCopy(draftKey, text)}
                            >
                                {#if copiedKey === draftKey}
                                    <SvgCheck class="h-4 w-4 text-success" />
                                {:else}
                                    <SvgCopy class="h-4 w-4" />
                                {/if}
                            </button>
                        {/if}
                    </div>
                {/each}
            </div>
        </div>
    {/if}

    <!-- History -->
    {#if captionData}
        {@const hasHistoryEntries = historyEntries.length > 0}
        <div>
            <button
                class="flex w-full cursor-pointer items-center justify-between"
                onclick={onHistoryToggle}
            >
                <h3 class="text-sm font-medium text-gray-300">History</h3>
                <span class="text-xs text-gray-500"
                    >{isHistoryExpanded ? '▲' : '▼'}{#if hasHistoryEntries}
                        ({historyEntries.length})
                    {/if}</span
                >
            </button>
            {#if isHistoryExpanded}
                {#if isLoadingHistory}
                    <SpinnerBlock size="h-4 w-4" label="Loading history..." />
                {:else if hasHistoryEntries}
                    <div class="mt-2 space-y-3">
                        {#each historyEntries as entry (entry.index)}
                            {@const hasExtras = Object.keys(entry.extras).length > 0}
                            <div class="rounded-lg bg-gray-800 p-3">
                                <div class="mb-1 flex items-center justify-between">
                                    <span class="text-xs font-medium text-gray-400"
                                        >Revision #{entry.index}</span
                                    >
                                    <button
                                        class="btn-secondary px-2 py-0.5 text-xs"
                                        onclick={() => handleRestoreClick(entry.index)}
                                        disabled={isRestoring}
                                    >
                                        {isRestoring ? 'Restoring...' : 'Restore'}
                                    </button>
                                </div>
                                <pre
                                    class="max-h-32 overflow-y-auto font-mono text-sm whitespace-pre-wrap text-gray-200">{entry.caption ||
                                        '(empty)'}</pre>
                                {#if hasExtras}
                                    <pre
                                        class="mt-2 max-h-32 overflow-y-auto rounded bg-gray-900 p-2 text-xs text-gray-400">{JSON.stringify(
                                            entry.extras,
                                            null,
                                            2
                                        )}</pre>
                                {/if}
                            </div>
                        {/each}
                    </div>
                {:else}
                    <p class="mt-2 text-sm text-gray-500 italic">No history</p>
                {/if}
            {/if}
        </div>
    {/if}
</div>
