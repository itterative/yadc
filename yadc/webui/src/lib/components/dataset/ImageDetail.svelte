<script lang="ts">
    import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import PromptPreview from '$lib/components/ui/PromptPreview.svelte';
    import {
        deleteDatasetItems,
        fetchCaption,
        fetchHistory,
        mediaUrl,
        restoreHistory,
        updateCaption,
        updateExtras,
        type CaptionData,
        type HistoryEntry,
        type ImageInfo
    } from '$lib/stores/datasetImages';
    import { captionSingleImage as startSingleCaptioning } from '$lib/stores/captionActions';
    import { currentlyCaptioning, getStoredCaption, clearStoredCaption } from '$lib/stores/events';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { friendlyErrorMessage } from '$lib/api';
    import { PasswordPromptCancelled } from '$lib/stores/passwordPrompt';
    import { confirmDialog } from '$lib/stores/confirm';

    interface Props {
        datasetName: string;
        item: ImageInfo | null;
        source?: 'upload' | 'import' | 'create';
        ondelete?: () => void;
    }

    let { datasetName, item, source, ondelete }: Props = $props();

    let captionData: CaptionData | null = $state(null);
    let isLoadingCaption = $state(false);
    let captionError: string | null = $state(null);
    let isEditing = $state(false);
    let editCaption = $state('');
    let isSaving = $state(false);

    // Captioning error state (spinner is controlled by store-derived isCaptioning)
    let captioningError: string | null = $state(null);

    // Derive captioning state from the global SSE store
    let isCaptioning = $derived(
        item !== null &&
            $currentlyCaptioning?.dataset_name === datasetName &&
            $currentlyCaptioning?.image_id === item.id
    );

    // Track previous isCaptioning to detect completion and reload caption
    let wasCaptioning = $state(false);

    // History state
    let historyEntries: HistoryEntry[] = $state([]);
    let isHistoryExpanded = $state(false);
    let isLoadingHistory = $state(false);
    let isRestoring = $state(false);

    // TOML extras editing state
    let isEditingExtras = $state(false);
    let editExtrasRaw = $state('');

    let imgElement: HTMLImageElement | null = $state(null);

    // Load caption when item changes
    $effect(() => {
        if (item === null) {
            captionData = null;
            captionError = null;
            isEditing = false;
            isEditingExtras = false;
            isHistoryExpanded = false;
            historyEntries = [];
            return;
        }

        let cancelled = false;
        isLoadingCaption = true;
        captionError = null;
        captionData = null;
        isEditing = false;
        isEditingExtras = false;
        isHistoryExpanded = false;
        historyEntries = [];

        (async () => {
            try {
                const data = await fetchCaption(datasetName, item.id);
                if (cancelled) {
                    return;
                }
                captionData = data;
                editCaption = data.caption || '';
                clearStoredCaption(item.id);
            } catch (e) {
                if (cancelled) {
                    return;
                }
                captionError = friendlyErrorMessage(e, 'Failed to load caption');
            } finally {
                if (!cancelled) {
                    isLoadingCaption = false;
                }
            }
        })();

        return () => {
            cancelled = true;
        };
    });

    let imgSrc = $derived(item !== null ? mediaUrl(datasetName, item.id) : '');

    async function handleSave() {
        if (item === null || !isEditing) {
            return;
        }
        isSaving = true;
        try {
            await updateCaption(datasetName, item.id, editCaption);
            captionData = { ...captionData!, caption: editCaption };
            isEditing = false;
            // Refresh history since save creates a new entry
            if (isHistoryExpanded) {
                try {
                    historyEntries = await fetchHistory(datasetName, item.id);
                } catch {
                    /* best effort */
                }
            }
        } catch (e) {
            captionError = friendlyErrorMessage(e, 'Failed to save caption');
        } finally {
            isSaving = false;
        }
    }

    function handleCancelEdit() {
        isEditing = false;
        editCaption = captionData?.caption || '';
    }

    async function handleCaptionImage() {
        if (item === null) {
            return;
        }
        captioningError = null;
        try {
            await startSingleCaptioning(datasetName, item.id);
        } catch (e) {
            if (e instanceof PasswordPromptCancelled) {
                return;
            }
            captioningError = friendlyErrorMessage(e, 'Failed to caption image');
        }
    }

    // Reload caption when a background single-image job completes
    $effect(() => {
        const current = isCaptioning;
        if (wasCaptioning && !current && item !== null) {
            const stored = getStoredCaption(item.id);
            if (stored !== undefined && captionData !== null) {
                captionData = { ...captionData, caption: stored };
                editCaption = stored;
                clearStoredCaption(item.id);
            } else {
                isLoadingCaption = true;
                captionError = null;
                (async () => {
                    try {
                        const data = await fetchCaption(datasetName, item.id);
                        captionData = data;
                        editCaption = data.caption || '';
                    } catch (e) {
                        captionError = friendlyErrorMessage(e, 'Failed to load caption');
                    } finally {
                        isLoadingCaption = false;
                    }
                })();
            }
        }
        wasCaptioning = current;
    });

    let isDeleting = $state(false);

    async function handleDeleteImage() {
        if (item === null || source !== 'upload' || !item.delete_path) {
            return;
        }
        const ok = await confirmDialog.danger(
            `Delete "${item.file_name}" and its sidecars? This cannot be undone.`
        );
        if (!ok) {
            return;
        }
        isDeleting = true;
        try {
            await deleteDatasetItems(datasetName, [item.delete_path]);
            ondelete?.();
        } catch (e) {
            captionError = friendlyErrorMessage(e, 'Failed to delete image');
        } finally {
            isDeleting = false;
        }
    }

    let isSavingExtras = $state(false);

    async function handleSaveExtras() {
        if (item === null || !isEditingExtras) {
            return;
        }
        isSavingExtras = true;
        try {
            await updateExtras(datasetName, item.id, editExtrasRaw);
            captionData = { ...captionData!, extras_raw: editExtrasRaw };
            isEditingExtras = false;
            // Refresh history since save creates a new entry
            if (isHistoryExpanded) {
                try {
                    historyEntries = await fetchHistory(datasetName, item.id);
                } catch {
                    /* best effort */
                }
            }
        } catch (e) {
            captionError = friendlyErrorMessage(e, 'Failed to save extras');
        } finally {
            isSavingExtras = false;
        }
    }

    function handleCancelEditExtras() {
        isEditingExtras = false;
        editExtrasRaw = captionData?.extras_raw || '';
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

{#if item !== null}
    <div class="flex h-full flex-col">
        <!-- Scrollable content -->
        <div class="flex-1 space-y-4 overflow-y-auto p-4">
            <!-- Filename -->
            <h2 class="dialog-title truncate text-base">{item.file_name}</h2>
            {#if source !== 'upload'}
                <p class="-mt-3 truncate text-xs text-gray-500">{item.path}</p>
            {/if}
            <!-- Image -->
            <div class="flex shrink-0 items-start justify-center">
                <img
                    src={imgSrc}
                    alt={item.file_name}
                    class="max-h-[50vh] w-auto rounded-lg bg-gray-500"
                    width={item.width || undefined}
                    height={item.height || undefined}
                    bind:this={imgElement}
                />
            </div>

            <!-- Dimensions & badges -->
            <div class="flex flex-wrap items-center gap-3">
                {#if item.width && item.height}
                    <span class="text-sm text-gray-400">{item.width}×{item.height}</span>
                {/if}
                {#if item.has_caption}
                    <span class="badge-success rounded-full px-2 py-1">Captioned</span>
                {:else}
                    <span class="badge-muted rounded-full px-2 py-1">No caption</span>
                {/if}
                {#if item.has_toml}
                    <span class="badge-accent rounded-full px-2 py-1">TOML</span>
                {/if}
                {#if item.draft_names.length > 0}
                    <span class="badge-error rounded-full px-2 py-1">
                        {item.draft_names.length} draft{item.draft_names.length !== 1 ? 's' : ''}
                    </span>
                {/if}
            </div>

            <!-- Caption -->
            <div>
                <div class="mb-2 flex items-center justify-between">
                    <h3 class="text-sm font-medium text-gray-300">Caption</h3>
                    {#if captionData && !isEditing && !isCaptioning}
                        <div class="flex items-center gap-3">
                            {#if !isCaptioning}
                                <button
                                    class="cursor-pointer text-xs text-accent hover:text-accent-hover"
                                    onclick={handleCaptionImage}
                                >
                                    Caption
                                </button>
                            {/if}
                            <button
                                class="cursor-pointer text-xs text-accent hover:text-accent-hover"
                                onclick={() => {
                                    editCaption = captionData?.caption || '';
                                    isEditing = true;
                                }}
                            >
                                Edit
                            </button>
                            {#if source === 'upload'}
                                <button
                                    class="cursor-pointer text-xs text-error hover:text-error/80"
                                    onclick={handleDeleteImage}
                                    disabled={isDeleting}
                                >
                                    {isDeleting ? 'Deleting…' : 'Delete'}
                                </button>
                            {/if}
                        </div>
                    {/if}
                </div>

                {#if isLoadingCaption}
                    <SpinnerBlock size="h-4 w-4" label="Loading caption..." />
                {:else if captionError}
                    <p class="text-sm text-error">{captionError}</p>
                {:else}
                    <div class="relative">
                        {#if isCaptioning}
                            <div
                                class="absolute inset-0 z-10 flex items-center justify-center gap-2 rounded-lg bg-black/60 text-sm text-gray-300"
                            >
                                <SvgSpinner class="h-4 w-4 animate-spin" />
                                <span>Generating caption…</span>
                            </div>
                        {:else if captioningError}
                            <div
                                class="absolute inset-x-0 top-0 z-10 rounded-t-lg bg-error/90 px-3 py-1.5 text-center text-sm text-white"
                            >
                                {captioningError}
                            </div>
                        {/if}
                        {#if isEditing}
                            <div class="space-y-2">
                                <textarea
                                    bind:value={editCaption}
                                    class="max-h-60 w-full resize-none rounded-lg bg-gray-800 p-3 font-mono text-sm whitespace-pre-wrap text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
                                    placeholder="Enter caption..."
                                    oninput={(e) => {
                                        const el = e.currentTarget;
                                        el.style.height = 'auto';
                                        el.style.height = Math.min(el.scrollHeight, 360) + 'px';
                                    }}
                                    use:autosize
                                ></textarea>
                                <div class="btn-bar">
                                    <button
                                        class="btn-secondary px-3 py-1.5"
                                        onclick={handleCancelEdit}
                                        disabled={isSaving}
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        class="btn-primary px-3 py-1.5"
                                        onclick={handleSave}
                                        disabled={isSaving}
                                    >
                                        {isSaving ? 'Saving...' : 'Save'}
                                    </button>
                                </div>
                            </div>
                        {:else if captionData}
                            {#if captionData.caption}
                                <pre
                                    class="max-h-60 overflow-y-auto rounded-lg bg-gray-800 p-3 text-sm whitespace-pre-wrap text-gray-200">{captionData.caption}</pre>
                            {:else}
                                <pre
                                    class="rounded-lg bg-gray-800 p-3 text-sm whitespace-pre-wrap text-gray-500 italic">No caption</pre>
                            {/if}
                        {/if}
                    </div>
                {/if}
            </div>

            <!-- History -->
            {#if captionData}
                {@const hasHistoryEntries = historyEntries.length > 0}
                <div>
                    <button
                        class="flex w-full cursor-pointer items-center justify-between"
                        onclick={async () => {
                            isHistoryExpanded = !isHistoryExpanded;
                            if (isHistoryExpanded && historyEntries.length === 0 && item !== null) {
                                isLoadingHistory = true;
                                try {
                                    historyEntries = await fetchHistory(datasetName, item.id);
                                } catch (e) {
                                    captionError = friendlyErrorMessage(
                                        e,
                                        'Failed to load history'
                                    );
                                } finally {
                                    isLoadingHistory = false;
                                }
                            }
                        }}
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
                                                onclick={async () => {
                                                    const ok = await confirmDialog.warning(
                                                        'Restore this revision? The current caption and extras will be saved to history first.'
                                                    );
                                                    if (!ok) {
                                                        return;
                                                    }
                                                    isRestoring = true;
                                                    try {
                                                        await restoreHistory(
                                                            datasetName,
                                                            item!.id,
                                                            entry.index
                                                        );
                                                        // Reload caption and history
                                                        const [caption, hist] = await Promise.all([
                                                            fetchCaption(datasetName, item!.id),
                                                            fetchHistory(datasetName, item!.id)
                                                        ]);
                                                        captionData = caption;
                                                        editCaption = caption.caption || '';
                                                        historyEntries = hist;
                                                    } catch (e) {
                                                        captionError = friendlyErrorMessage(
                                                            e,
                                                            'Failed to restore'
                                                        );
                                                    } finally {
                                                        isRestoring = false;
                                                    }
                                                }}
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

            <!-- TOML extras -->
            {#if captionData?.extras_raw}
                <div>
                    <div class="mb-2 flex items-center justify-between">
                        <h3 class="text-sm font-medium text-gray-300">TOML Extras</h3>
                        {#if !isEditingExtras}
                            <button
                                class="cursor-pointer text-xs text-accent hover:text-accent-hover"
                                onclick={() => {
                                    editExtrasRaw = captionData?.extras_raw || '';
                                    isEditingExtras = true;
                                }}
                            >
                                Edit
                            </button>
                        {:else}
                            <div class="flex gap-2">
                                <button
                                    class="btn-secondary px-2 py-1 text-xs"
                                    onclick={handleCancelEditExtras}
                                    disabled={isSavingExtras}
                                >
                                    Cancel
                                </button>
                                <button
                                    class="btn-primary px-2 py-1 text-xs"
                                    onclick={handleSaveExtras}
                                    disabled={isSavingExtras}
                                >
                                    {isSavingExtras ? 'Saving...' : 'Save'}
                                </button>
                            </div>
                        {/if}
                    </div>
                    <TomlEditor
                        class="text-sm"
                        value={isEditingExtras ? editExtrasRaw : captionData.extras_raw}
                        editable={isEditingExtras}
                        onchange={(v) => (editExtrasRaw = v)}
                    />
                </div>
            {:else if captionData?.extras && Object.keys(captionData.extras).length > 0}
                <div>
                    <h3 class="mb-2 text-sm font-medium text-gray-300">TOML Extras</h3>
                    <pre
                        class="overflow-x-auto rounded-lg bg-gray-800 p-3 text-xs text-gray-400">{JSON.stringify(
                            captionData.extras,
                            null,
                            2
                        )}</pre>
                </div>
            {/if}

            <!-- Drafts -->
            {#if captionData?.drafts && Object.keys(captionData.drafts).length > 0}
                <div>
                    <h3 class="mb-2 text-sm font-medium text-gray-300">Drafts</h3>
                    <div class="space-y-2">
                        {#each Object.entries(captionData.drafts) as [name, text] (name)}
                            <div class="rounded-lg bg-gray-800 p-3">
                                <p class="mb-1 text-xs font-medium text-accent">{name}</p>
                                <pre
                                    class="max-h-40 overflow-y-auto font-mono text-sm whitespace-pre-wrap text-gray-200">{text}</pre>
                            </div>
                        {/each}
                    </div>
                </div>
            {/if}

            <!-- Prompt Preview -->
            <PromptPreview {datasetName} imageId={item.id} />
        </div>
    </div>
{/if}
