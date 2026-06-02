<script lang="ts">
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
    import {
        captionSingleImage as startSingleCaptioning,
        stopCaptioning
    } from '$lib/stores/captionActions';
    import { currentlyCaptioning, getStoredCaption, clearStoredCaption } from '$lib/stores/events';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import SvgSparkle from '$lib/icons/SvgSparkle.svelte';
    import SvgVisibility from '$lib/icons/SvgVisibility.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import CompactPillTabs from '$lib/components/ui/tabs/CompactPillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import { confirmDialog } from '$lib/stores/confirm';
    import { toast } from '$lib/stores/toasts';
    import { friendlyErrorMessage } from '$lib/api';
    import Caption from './Caption.svelte';
    import Preview from './Preview.svelte';
    import ExtrasTab from './Extras.svelte';

    interface Props {
        datasetName: string;
        item: ImageInfo;
        source?: 'upload' | 'import' | 'create';
        ondelete?: () => void;
    }

    let { datasetName, item, source, ondelete }: Props = $props();

    let captionData: CaptionData | null = $state(null);
    let isLoadingCaption = $state(false);
    let captionError: string | null = $state(null);
    let historyEntries: HistoryEntry[] = $state([]);
    let isLoadingHistory = $state(false);
    let isRestoring = $state(false);
    let isDeleting = $state(false);
    let isSavingExtras = $state(false);
    let isHistoryExpanded = $state(false);
    let copiedKey: string | null = $state(null);
    let copyTimeout: ReturnType<typeof setTimeout> | null = null;
    let wasCaptioning = $state(false);
    let activeTab = $state('caption');

    let isCaptioning = $derived(
        item !== null &&
            $currentlyCaptioning?.dataset_name === datasetName &&
            $currentlyCaptioning?.image_id === item.id
    );

    let imgSrc = $derived(item !== null ? mediaUrl(datasetName, item.id) : '');

    // Load caption when item changes
    $effect(() => {
        if (item === null) {
            captionData = null;
            captionError = null;
            historyEntries = [];
            isHistoryExpanded = false;
            return;
        }

        let cancelled = false;
        const id = item.id;
        const dsName = datasetName;
        isLoadingCaption = true;
        captionError = null;
        captionData = null;
        historyEntries = [];
        isHistoryExpanded = false;

        (async () => {
            try {
                const data = await fetchCaption(dsName, id);
                if (cancelled) {
                    return;
                }
                captionData = data;
                clearStoredCaption(id);
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

    // Re-fetch caption + history when a single-image captioning job completes
    $effect(() => {
        const current = isCaptioning;
        if (wasCaptioning && !current && item !== null) {
            const id = item.id;
            const dsName = datasetName;
            // Show the stored caption immediately for instant feedback
            const stored = getStoredCaption(id);
            if (stored !== undefined && captionData !== null) {
                captionData = { ...captionData, caption: stored };
                clearStoredCaption(id);
            }
            (async () => {
                try {
                    const [data, hist] = await Promise.all([
                        fetchCaption(dsName, id),
                        fetchHistory(dsName, id)
                    ]);
                    if (item?.id !== id) {
                        return;
                    }
                    captionData = data;
                    historyEntries = hist;
                } catch (e) {
                    captionError = friendlyErrorMessage(e, 'Failed to load caption');
                }
            })();
        }
        wasCaptioning = current;
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

    async function handleHistoryToggle() {
        isHistoryExpanded = !isHistoryExpanded;
        if (isHistoryExpanded && historyEntries.length === 0 && item !== null) {
            isLoadingHistory = true;
            try {
                historyEntries = await fetchHistory(datasetName, item.id);
            } catch (e) {
                captionError = friendlyErrorMessage(e, 'Failed to load history');
            } finally {
                isLoadingHistory = false;
            }
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

<div class="grid grid-cols-1 grid-rows-[min-content_minmax(0,1fr)] min-h-full h-fit flex-col p-4">
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
                <span class="text-sm italic text-muted ml-auto mr-2">{item.width}×{item.height}</span>
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
    <CompactPillTabs bind:value={activeTab} class="h-full pt-4">
        <Tab
            id="caption"
            label="Caption"
            icon={SvgSparkle}
            class="h-full"
        >
            <Caption
                {item}
                {captionData}
                {isLoadingCaption}
                {captionError}
                {isCaptioning}
                {historyEntries}
                {isLoadingHistory}
                {isRestoring}
                {isHistoryExpanded}
                {copiedKey}
                onCaptioningStart={handleCaptioningStart}
                onCaptioningCancel={handleCaptioningCancel}
                onSaveCaption={handleSaveCaption}
                onRestoreHistory={handleRestoreHistory}
                onHistoryToggle={handleHistoryToggle}
                onCopy={handleCopy}
            />
        </Tab>
        <Tab
            id="preview"
            label="Preview"
            icon={SvgVisibility}
            class="h-full"
        >
            <Preview {datasetName} {captionData} {item} />
        </Tab>
        <Tab id="extras" label="Extras" icon={SvgEdit} class="h-full overflow-y-auto pt-0">
            <ExtrasTab {item} {captionData} {isSavingExtras} onSaveExtras={handleSaveExtras} />
        </Tab>
    </CompactPillTabs>
</div>
