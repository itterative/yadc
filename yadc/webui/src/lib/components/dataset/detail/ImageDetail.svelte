<script lang="ts">
    import {
        deleteDatasetItems,
        deleteDraft as deleteDraftApi,
        deleteHistory as deleteHistoryApi,
        fetchCaption,
        fetchHistory,
        mediaUrl,
        restoreHistory,
        updateCaption,
        updateExtras,
        writeDraft as writeDraftApi,
        type CaptionData,
        type HistoryEntry,
        type ImageInfo
    } from '$lib/stores/dataset';
    import {
        captionSingleImage as startSingleCaptioning,
        currentlyCaptioning,
        stopCaptioning
    } from '$lib/stores/caption';
    import { lastCaptionedImage, type ImageCaptionedEvent } from '$lib/stores/events';
    import { untrack } from 'svelte';
    import { get } from 'svelte/store';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import SvgSparkle from '$lib/icons/SvgSparkle.svelte';
    import SvgVisibility from '$lib/icons/SvgVisibility.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgTag from '$lib/icons/SvgTag.svelte';
    import CompactPillTabs from '$lib/components/ui/tabs/CompactPillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import { confirmDialog } from '$lib/stores/confirm';
    import { toast } from '$lib/stores/toasts';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, linkedController } from '$lib/abort';
    import Caption from './Caption.svelte';
    import Preview from './Preview.svelte';
    import ExtrasTab from './Extras.svelte';
    import TagsTab from './Tags.svelte';

    interface Props {
        datasetName: string;
        item: ImageInfo;
        source?: 'upload' | 'import' | 'create';
        ondelete?: () => void;
    }

    let { datasetName, item, source, ondelete }: Props = $props();

    // Component-level abort scope — linked to any parent context and
    // propagated to children (e.g. PromptPreview). Aborting this controller
    // cancels all in-flight requests in this component and its descendants.
    const abort = createAbortContext();

    let captionData: CaptionData | null = $state(null);
    let isLoadingCaption = $state(false);
    let captionError: string | null = $state(null);
    let historyEntries: HistoryEntry[] = $state([]);
    let isRestoring = $state(false);
    let isDeleting = $state(false);
    let isSavingExtras = $state(false);
    let copiedKey: string | null = $state(null);
    let copyTimeout: ReturnType<typeof setTimeout> | null = null;
    let activeTab = $state('caption');

    // Last image_captioned event already handled. Plain (non-reactive) so
    // updating it never re-triggers the refresh effect; initialized to the
    // current event so a caption that arrived before focus doesn't refresh.
    let handledCaptionedEvent: ImageCaptionedEvent | null = get(lastCaptionedImage);

    let isCaptioning = $derived.by(() => {
        if (item === null) {
            return false;
        }
        for (const entry of $currentlyCaptioning) {
            if (entry.dataset_name === datasetName && entry.image_id === item.id) {
                return true;
            }
        }
        return false;
    });

    let imgSrc = $derived(item !== null ? mediaUrl(datasetName, item.id) : '');

    // Load caption when item changes
    $effect(() => {
        if (item === null) {
            captionData = null;
            captionError = null;
            historyEntries = [];
            return;
        }

        const id = item.id;
        const dsName = datasetName;
        isLoadingCaption = true;
        captionError = null;
        captionData = null;
        historyEntries = [];

        // Effect-scoped controller that also aborts if the component-level
        // scope fires (e.g. parent unmount / parent abort).
        const controller = linkedController(abort.signal);

        (async () => {
            try {
                const [data, hist] = await Promise.all([
                    fetchCaption(dsName, id, controller.signal),
                    fetchHistory(dsName, id, controller.signal)
                ]);
                if (controller.signal.aborted) {
                    return;
                }
                captionData = data;
                historyEntries = hist;
            } catch (e) {
                if (controller.signal.aborted) {
                    return;
                }
                captionError = friendlyErrorMessage(e, 'Failed to load caption');
            } finally {
                if (!controller.signal.aborted) {
                    isLoadingCaption = false;
                }
            }
        })();

        return () => controller.abort();
    });

    // Re-fetch caption + history when captioning finishes for the focused
    // image. We react to the per-image "captioned" event directly (matching
    // the selected image) instead of inferring completion from the in-flight
    // store, which would require a state transition that is easy to get wrong.
    //
    // The optimistic swap reads captionData via untrack() so the assignment
    // can't re-trigger this effect (a self-retrigger would run the cleanup and
    // abort the fetch before its response arrives). handledCaptionedEvent is a
    // plain non-reactive var so we refresh only for an event arriving while
    // this image is focused, not one already present at mount.
    $effect(() => {
        const event = $lastCaptionedImage;
        if (event === null || event === handledCaptionedEvent) {
            return;
        }
        handledCaptionedEvent = event;
        if (event.dataset_name !== datasetName || event.id !== item.id) {
            return;
        }

        const id = item.id;
        const dsName = datasetName;
        const controller = linkedController(abort.signal);

        // Optimistic: swap in the freshly-captioned text for instant feedback
        // while the re-fetch is in flight.
        untrack(() => {
            if (captionData !== null) {
                captionData = { ...captionData, caption: event.caption };
            }
        });

        (async () => {
            try {
                const [data, hist] = await Promise.all([
                    fetchCaption(dsName, id, controller.signal),
                    fetchHistory(dsName, id, controller.signal)
                ]);
                if (controller.signal.aborted || item?.id !== id) {
                    return;
                }
                captionData = data;
                historyEntries = hist;
            } catch (e) {
                if (controller.signal.aborted) {
                    return;
                }
                captionError = friendlyErrorMessage(e, 'Failed to load caption');
            }
        })();

        return () => controller.abort();
    });

    async function handleSaveCaption(text: string) {
        if (item === null) {
            return;
        }
        try {
            await updateCaption(datasetName, item.id, text);
            const [data, hist] = await Promise.all([
                fetchCaption(datasetName, item.id),
                fetchHistory(datasetName, item.id)
            ]);
            captionData = data;
            historyEntries = hist;
        } catch (e) {
            captionError = friendlyErrorMessage(e, 'Failed to save caption');
            throw e;
        }
    }

    async function handleCaptioningStart() {
        if (item === null) {
            return;
        }
        await startSingleCaptioning(datasetName, item.id);
    }

    async function handleCaptioningCancel() {
        if (item === null) {
            return;
        }
        await stopCaptioning(datasetName);
    }

    async function handleRestoreHistory(index: number) {
        if (item === null) {
            return;
        }
        isRestoring = true;
        try {
            await restoreHistory(datasetName, item.id, index);
            const [data, hist] = await Promise.all([
                fetchCaption(datasetName, item.id),
                fetchHistory(datasetName, item.id)
            ]);
            captionData = data;
            historyEntries = hist;
        } catch (e) {
            captionError = friendlyErrorMessage(e, 'Failed to restore');
            throw e;
        } finally {
            isRestoring = false;
        }
    }

    async function handleDeleteHistory(entryHash: string) {
        if (item === null) {
            return;
        }
        const ok = await confirmDialog.danger(
            'Delete this history revision? This cannot be undone.'
        );
        if (!ok) {
            return;
        }
        try {
            await deleteHistoryApi(datasetName, item.id, entryHash);
            historyEntries = await fetchHistory(datasetName, item.id);
        } catch (e) {
            captionError = friendlyErrorMessage(e, 'Failed to delete history entry');
        }
    }

    async function handleDeleteDraft(draftName: string) {
        if (item === null) {
            return;
        }
        const ok = await confirmDialog.danger(
            `Delete draft "${draftName}"? This cannot be undone.`
        );
        if (!ok) {
            return;
        }
        try {
            await deleteDraftApi(datasetName, item.id, draftName);
            const data = await fetchCaption(datasetName, item.id);
            captionData = data;
        } catch (e) {
            captionError = friendlyErrorMessage(e, 'Failed to delete draft');
        }
    }

    async function handleWriteDraft(draftName: string, text: string) {
        if (item === null) {
            return;
        }
        try {
            await writeDraftApi(datasetName, item.id, draftName, text);
            const data = await fetchCaption(datasetName, item.id);
            captionData = data;
        } catch (e) {
            captionError = friendlyErrorMessage(e, 'Failed to save draft');
            throw e;
        }
    }

    async function handleSaveExtras(extrasRaw: string) {
        if (item === null) {
            return;
        }
        isSavingExtras = true;
        try {
            await updateExtras(datasetName, item.id, extrasRaw);
            const [data, hist] = await Promise.all([
                fetchCaption(datasetName, item.id),
                fetchHistory(datasetName, item.id)
            ]);
            captionData = data;
            historyEntries = hist;
        } catch (e) {
            // Show a toast because the user is in the Edit tab and the inline
            // captionError is only visible in the Caption tab.
            toast.error(friendlyErrorMessage(e, 'Failed to save extras'));
            throw e;
        } finally {
            isSavingExtras = false;
        }
    }

    /** Refresh caption data after the Tags tab writes a draft / extras so the
     *  Caption (drafts) and Extras tabs reflect the change. */
    async function handleTagsSaved() {
        if (item === null) {
            return;
        }
        try {
            const [data, hist] = await Promise.all([
                fetchCaption(datasetName, item.id),
                fetchHistory(datasetName, item.id)
            ]);
            captionData = data;
            historyEntries = hist;
        } catch {
            // Non-fatal — the save itself succeeded; this is just a refresh.
        }
    }

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

    async function handleCopy(key: string, text: string) {
        if (!text) {
            return;
        }
        try {
            await navigator.clipboard.writeText(text);
            copiedKey = key;
            if (copyTimeout !== null) {
                clearTimeout(copyTimeout);
            }
            copyTimeout = setTimeout(() => {
                copiedKey = null;
            }, 1500);
        } catch {
            toast.warning('Failed to copy to clipboard');
        }
    }
</script>

<div class="grid h-fit min-h-full grid-cols-1 grid-rows-[min-content_minmax(0,1fr)] flex-col p-4">
    <!-- Header — always visible across all tabs -->
    <div class="space-y-4 pb-0">
        <!-- Filename -->
        <div class="flex items-center gap-2">
            <h2 class="dialog-title min-w-0 flex-1 truncate text-base">{item.file_name}</h2>
            {#if source === 'upload' && item.delete_path}
                <button
                    class="shrink-0 cursor-pointer rounded p-1.5 text-gray-500 transition-colors hover:bg-gray-800 hover:text-error disabled:cursor-not-allowed disabled:opacity-50"
                    onclick={handleDeleteImage}
                    disabled={isDeleting}
                    aria-label="Delete image"
                    title="Delete image"
                >
                    <SvgDelete class="h-5 w-5" />
                </button>
            {/if}
        </div>
        <!-- {#if source !== 'upload'} -->
        <p class="-mt-3 truncate text-xs text-gray-500">{item.delete_path ?? item.path}</p>
        <!-- {/if} -->
        <!-- Dimensions & badges -->
        <div class="flex flex-wrap items-center gap-3">
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
            {#if item.width && item.height}
                <span class="mr-2 ml-auto text-sm text-muted italic"
                    >{item.width}×{item.height}</span
                >
            {/if}
        </div>
        <!-- Image -->
        <div class="flex shrink-0 items-start justify-center">
            <img
                src={imgSrc}
                alt={item.file_name}
                class="max-h-[40vh] w-auto rounded-lg bg-gray-500"
                width={item.width || undefined}
                height={item.height || undefined}
            />
        </div>
    </div>

    <!-- Inner tabs: Caption / Preview / Edit -->
    <CompactPillTabs storageId="dataset/imageDetails" bind:value={activeTab} class="h-full pt-4">
        <Tab id="caption" label="Caption" icon={SvgSparkle} class="h-full">
            <Caption
                {datasetName}
                {item}
                {captionData}
                {isLoadingCaption}
                {captionError}
                {isCaptioning}
                {historyEntries}
                {isRestoring}
                {copiedKey}
                onCaptioningStart={handleCaptioningStart}
                onCaptioningCancel={handleCaptioningCancel}
                onSaveCaption={handleSaveCaption}
                onRestoreHistory={handleRestoreHistory}
                onDeleteHistory={handleDeleteHistory}
                onDeleteDraft={handleDeleteDraft}
                onPromoteDraft={handleSaveCaption}
                onWriteDraft={handleWriteDraft}
                onCopy={handleCopy}
            />
        </Tab>
        <Tab id="preview" label="Preview" icon={SvgVisibility} class="h-full">
            <Preview {datasetName} {captionData} {item} />
        </Tab>
        <Tab id="extras" label="Extras" icon={SvgEdit} class="h-full overflow-y-auto pt-0">
            <ExtrasTab {item} {captionData} {isSavingExtras} onSaveExtras={handleSaveExtras} />
        </Tab>
        <Tab id="tags" label="Tags" icon={SvgTag} class="h-full overflow-y-auto">
            <TagsTab {datasetName} {item} onTagsSaved={handleTagsSaved} />
        </Tab>
    </CompactPillTabs>
</div>
