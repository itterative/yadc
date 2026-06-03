<script lang="ts">
    import { page } from '$app/stores';
    import { onMount, untrack } from 'svelte';
    import { browser } from '$app/environment';
    import { get } from 'svelte/store';
    import DatasetBrowser from '$lib/components/dataset/browser/DatasetBrowser.svelte';
    import SidePanel from './SidePanel.svelte';
    import AddFilesDialog from './AddFilesDialog.svelte';
    import DropUploadZone from './DropUploadZone.svelte';
    import SvgChevronLeft from '$lib/icons/SvgChevronLeft.svelte';
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import SvgUpload from '$lib/icons/SvgUpload.svelte';
    import Alert from '$lib/components/ui/Alert.svelte';
    import Topbar from '$lib/components/ui/Topbar.svelte';
    import {
        captioningStatus,
        pendingDatasetChanges,
        clearPendingDatasetChange,
        registerJobId,
        resumptionFailed,
        clearResumptionFailed,
        setCaptioningStatus,
        lastCaptionedImage,
        lastCaptionError,
        currentlyCaptioning
    } from '$lib/stores/events';
    import { lastStartedJobId } from '$lib/stores/captionActions';
    import { toast, dismissToast } from '$lib/stores/toasts';
    import { friendlyErrorMessage } from '$lib/api';
    import {
        fetchDatasets,
        fetchImages,
        fetchCaptioningStatus,
        rescanDataset,
        type DatasetInfo,
        type ImageInfo
    } from '$lib/stores/datasetImages';

    let datasetName = $derived($page.params.name ?? '');

    let datasets: DatasetInfo[] = $state([]);
    let currentDataset: DatasetInfo | null = $state(null);
    let focusedItem: ImageInfo | null = $state(null);

    let images: ImageInfo[] = $state([]);
    let isLoading = $state(true);
    let isLoadingMore = $state(false);
    let hasMore = $state(true);
    let error: string | null = $state(null);

    let nextToken: string | null = null;
    const PAGE_SIZE = 50;

    // --- Mobile panel state ---
    let panelOpen = $state(false);

    // --- Captioning state ---

    // Derive batch captioning state from the global SSE store
    let isBatchCaptioning = $derived(
        $captioningStatus?.dataset_name === datasetName &&
            ($captioningStatus.status === 'running' || $captioningStatus.status === 'stopping')
    );

    // ID of the image currently being captioned (for pulsing animation)
    let captioningImageId = $derived(
        $currentlyCaptioning?.dataset_name === datasetName ? $currentlyCaptioning.image_id : null
    );

    let isStopping = $derived(
        $captioningStatus?.dataset_name === datasetName && $captioningStatus.status === 'stopping'
    );

    let captionPct = $derived(
        $captioningStatus?.total > 0
            ? Math.round(($captioningStatus.processed / $captioningStatus.total) * 100)
            : 0
    );

    // --- Drop-to-upload state (page-level) ---

    // Files captured by a drop event, used to pre-populate AddFilesDialog.
    let droppedFiles: File[] = $state([]);
    let addFilesDialogOpen = $state(false);

    // Whether uploads are allowed for the current dataset state.
    let canUpload = $derived.by(() => currentDataset?.source === 'upload' && !isBatchCaptioning);

    // Why uploads are blocked (drives the warning overlay text).
    let uploadBlockedReason = $derived.by(() => {
        if (isBatchCaptioning) {
            return 'Captioning in progress — uploads are disabled until it finishes';
        }
        if (currentDataset && currentDataset.source !== 'upload') {
            return 'External datasets do not support file uploads';
        }
        return null;
    });

    // Register job_id from incoming status events so dataset_changed events
    // from our own captioning are suppressed
    $effect(() => {
        const s = $captioningStatus;
        if (
            s?.dataset_name === datasetName &&
            s.job_id &&
            (s.status === 'running' || s.status === 'stopping')
        ) {
            registerJobId(s.job_id);
        }
    });

    // Fire a toast when the captioning job we started reaches a terminal state.
    // Tracking by job_id avoids race conditions between SSE events and the
    // HTTP response, and prevents toasting for stale jobs on page load.
    $effect(() => {
        const s = $captioningStatus;
        if (s?.dataset_name !== datasetName) {
            return;
        }
        if (s.status !== 'error' && s.status !== 'done' && s.status !== 'cancelled') {
            return;
        }
        if (s.job_id !== $lastStartedJobId) {
            return;
        }
        // Reset so navigating back to this page with a stale status
        // does not re-trigger the toast.
        lastStartedJobId.set('');
        handleCaptioningDone();
    });

    // --- Filesystem watcher state ---

    let watcherToastId: string | null = $state(null);
    let isRefreshing = $state(false);

    // Show a toast when external filesystem changes are detected.
    // Re-use the same toast id so rapid changes don't stack duplicates.
    $effect(() => {
        const name = datasetName;
        return pendingDatasetChanges.subscribe((set) => {
            if (set.has(name)) {
                if (!watcherToastId) {
                    watcherToastId = toast.info('Dataset files have changed', {
                        duration: 0,
                        actions: [
                            {
                                label: 'Refresh',
                                handler: () => {
                                    handleRefreshFromWatcher();
                                }
                            }
                        ]
                    });
                }
            } else {
                if (watcherToastId) {
                    dismissToast(watcherToastId);
                    watcherToastId = null;
                }
            }
        });
    });

    // Update individual grid tiles as captioning progresses.
    $effect(() => {
        const event = $lastCaptionedImage;
        if (!event || event.dataset_name !== datasetName) {
            return;
        }
        const now = Date.now();
        untrack(() => {
            images = images.map((img) =>
                img.id === event.id
                    ? {
                          ...img,
                          has_caption: event.has_caption,
                          has_toml: event.has_toml,
                          draft_names: event.draft_names,
                          caption_error: undefined,
                          flash: now
                      }
                    : img
            );
        });
    });

    // Mark tiles that failed captioning with an error indicator.
    $effect(() => {
        const event = $lastCaptionError;
        if (!event || event.dataset_name !== datasetName) {
            return;
        }
        const now = Date.now();
        untrack(() => {
            images = images.map((img) =>
                img.id === event.image_id ? { ...img, caption_error: event.error, flash: now } : img
            );
        });
    });

    async function loadInitial(name: string) {
        nextToken = null;
        isLoading = true;
        error = null;

        try {
            const page = await fetchImages(name, { limit: PAGE_SIZE });
            images = page.images;
            nextToken = page.next_token;
            hasMore = page.next_token !== null;
            // We just loaded a fresh view — clear any stale pending-change flag
            clearPendingDatasetChange(name);
        } catch (e) {
            error = friendlyErrorMessage(e, 'Failed to load images');
            toast.error(`Failed to load images: ${error}`);
        } finally {
            isLoading = false;
        }
    }

    async function loadMore() {
        if (!datasetName) {
            return;
        }
        if (isLoadingMore || !hasMore || nextToken === null) {
            return;
        }
        isLoadingMore = true;

        try {
            const page = await fetchImages(datasetName, {
                limit: PAGE_SIZE,
                afterId: parseInt(nextToken)
            });
            images = [...images, ...page.images];
            nextToken = page.next_token;
            hasMore = page.next_token !== null;
        } catch (e) {
            error = friendlyErrorMessage(e, 'Failed to load more images');
        } finally {
            isLoadingMore = false;
        }
    }

    // React to dataset name changes
    $effect(() => {
        const name = datasetName;
        if (!browser || !name) {
            return;
        }
        loadInitial(name);
        // Poll current captioning status to handle mid-captioning page loads.
        // Only seed the store if it is idle for this dataset so we do not
        // clobber a more recent SSE event.
        (async () => {
            try {
                const status = await fetchCaptioningStatus(name);
                const current = get(captioningStatus);
                if (
                    status.dataset_name === name &&
                    (current.status === 'idle' || current.dataset_name !== name)
                ) {
                    setCaptioningStatus(status);
                }
            } catch {
                /* ignore — SSE will eventually provide status */
            }
        })();
    });

    // Find the dataset info from the list
    $effect(() => {
        if (datasets.length > 0 && datasetName) {
            currentDataset = datasets.find((d) => d.name === datasetName) ?? null;
        }
    });

    onMount(() => {
        if (!browser) {
            return;
        }
        (async () => {
            try {
                datasets = await fetchDatasets();
            } catch {
                datasets = [];
            }
        })();
    });

    // --- Side panel state ---

    let panelTab: 'caption' | 'details' | 'config' = $state('caption');

    function handleItemClick(item: ImageInfo) {
        focusedItem = item;
        panelTab = 'details';
        panelOpen = true;
    }

    function handlePanelClose() {
        focusedItem = null;
        panelTab = 'caption';
        panelOpen = false;
    }

    function handleImageDelete() {
        focusedItem = null;
        panelTab = 'caption';
        panelOpen = false;
        if (browser && datasetName) {
            loadInitial(datasetName);
            (async () => {
                try {
                    datasets = await fetchDatasets();
                } catch {
                    /* ignore */
                }
            })();
        }
    }

    function handleCaptioningDone() {
        const status = get(captioningStatus);
        if (status.dataset_name !== datasetName) {
            return;
        }

        // Only refresh the grid if images were actually modified.
        // Fast failures (e.g. env decryption error with 0 processed)
        // leave the dataset untouched, so reloading is pointless.
        const hadProgress = status.processed > 0 || status.errors > 0;
        if (browser && datasetName && hadProgress) {
            // If SSE resumption failed we may have missed per-image events,
            // so reload page 1 as a safety net. Otherwise the tiles are
            // already current from live ImageCaptionedEvents.
            if (get(resumptionFailed)) {
                toast.warning(
                    'Some captioning events were missed. Refreshing grid to ensure accuracy.'
                );
                loadInitial(datasetName);
            }
            (async () => {
                try {
                    datasets = await fetchDatasets();
                } catch {
                    /* ignore */
                }
            })();
        }

        if (status.status === 'cancelled') {
            toast.info(`Captioning cancelled (${status.processed}/${status.total} processed)`);
        } else if (status.status === 'error') {
            const details =
                status.error_messages.length > 0
                    ? status.error_messages
                    : [status.error ?? 'Unknown error'];
            toast.error('Captioning failed', { details });
        } else if (status.errors > 0) {
            const details = status.error_messages.slice(0, 2);
            toast.warning(
                `Captioning finished with ${status.errors} error${status.errors === 1 ? '' : 's'} (${status.processed}/${status.total} processed)`,
                { details }
            );
        }
    }

    async function handleManualRefresh() {
        if (!browser || !datasetName || isRefreshing) {
            return;
        }
        isRefreshing = true;
        try {
            await rescanDataset(datasetName);
            // Dismiss any pending watcher toast since we've just refreshed
            if (watcherToastId) {
                dismissToast(watcherToastId);
                watcherToastId = null;
            }
            clearPendingDatasetChange(datasetName);
            await loadInitial(datasetName);
            try {
                datasets = await fetchDatasets();
            } catch {
                /* ignore */
            }
        } catch (e) {
            toast.error(`Failed to refresh: ${friendlyErrorMessage(e, 'Unknown error')}`);
        } finally {
            isRefreshing = false;
        }
    }

    function handleRefreshFromWatcher() {
        if (!browser || !datasetName) {
            return;
        }
        // Dismiss the toast and clear state before reloading
        if (watcherToastId) {
            dismissToast(watcherToastId);
            watcherToastId = null;
        }
        clearPendingDatasetChange(datasetName);
        loadInitial(datasetName);
        (async () => {
            try {
                datasets = await fetchDatasets();
            } catch {
                /* ignore */
            }
        })();
    }

    // --- Drop-to-upload handlers ---

    function handleFilesDropped(files: File[]) {
        droppedFiles = files;
        addFilesDialogOpen = true;
    }

    function handleAddFilesClose() {
        addFilesDialogOpen = false;
        droppedFiles = [];
    }

    async function handleAddFilesComplete() {
        addFilesDialogOpen = false;
        droppedFiles = [];
        if (!browser || !datasetName) {
            return;
        }
        // Re-fetch images + dataset stats so the new files show up immediately.
        // The DatasetChangedEvent watcher will normally trigger a refresh toast,
        // but we already know the upload succeeded so we go straight to the
        // load path without showing the user an extra "files have changed" prompt.
        await loadInitial(datasetName);
        clearPendingDatasetChange(datasetName);
        if (watcherToastId) {
            dismissToast(watcherToastId);
            watcherToastId = null;
        }
        try {
            datasets = await fetchDatasets();
        } catch {
            /* ignore */
        }
    }

    function handleAddFilesClick() {
        droppedFiles = [];
        addFilesDialogOpen = true;
    }
</script>

<svelte:head>
    <title>{datasetName} — yadc</title>
</svelte:head>

<Topbar>
    <a href="#/" class="-ml-2 hidden p-2 text-gray-400 transition-colors hover:text-white md:flex">
        <SvgChevronLeft class="h-6 w-6" />
    </a>
    <div class="relative min-w-0 flex-1">
        <div class="flex items-center gap-3">
            <h1 class="overflow-hidden text-xl font-bold text-nowrap text-ellipsis text-white">
                {datasetName}
            </h1>
            {#if currentDataset}
                {#if currentDataset.source === 'upload'}
                    <span class="badge-muted badge-sm shrink-0">Managed</span>
                {:else}
                    <span class="badge-muted badge-sm shrink-0">External</span>
                {/if}
            {/if}
            {#if isBatchCaptioning}
                <SvgSpinner
                    class="h-4 w-4 shrink-0 animate-spin {isStopping
                        ? 'text-yellow-400'
                        : 'text-accent'}"
                />
            {/if}
        </div>
        <div class="relative mt-0.5 flex items-center">
            {#if isBatchCaptioning}
                {#if isStopping}
                    <span class="text-sm text-yellow-300">Stopping…</span>
                {:else}
                    <span class="text-sm text-gray-400">
                        Captioning… {$captioningStatus.processed}/{$captioningStatus.total}
                        ({captionPct}%)
                    </span>
                {/if}
            {:else if currentDataset}
                <p class="text-sm text-gray-400">
                    {currentDataset.image_count} images · {currentDataset.has_caption} captioned ·
                    {currentDataset.has_toml}
                    with TOML
                </p>
            {:else}
                <p class="text-sm text-gray-400">...</p>
            {/if}

            {#if isBatchCaptioning && !isStopping}
                <div class="absolute right-0 -bottom-1 left-0 h-0.5 bg-bg">
                    <div
                        class="h-full rounded-full bg-accent transition-all duration-300 ease-out"
                        style:width="{captionPct}%"
                    ></div>
                </div>
            {/if}
        </div>
    </div>
    {#if canUpload}
        <button
            class="shrink-0 cursor-pointer rounded-lg p-2 text-gray-400 transition-colors hover:bg-white/10 hover:text-white"
            title="Add files to this dataset"
            onclick={handleAddFilesClick}
        >
            <SvgUpload class="h-5 w-5" />
        </button>
    {/if}
    <button
        class="shrink-0 cursor-pointer rounded-lg p-2 text-gray-400 transition-colors hover:bg-white/10 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
        title="Refresh dataset from disk"
        disabled={isRefreshing}
        onclick={handleManualRefresh}
    >
        {#if isRefreshing}
            <SvgSpinner class="h-5 w-5 animate-spin" />
        {:else}
            <SvgRefresh class="h-5 w-5" />
        {/if}
    </button>
</Topbar>

<div class="flex h-full flex-col gap-4">
    <!-- SSE resumption failure notification -->
    <Alert
        variant="warning"
        visible={$resumptionFailed}
        dismissable
        ondismiss={clearResumptionFailed}
    >
        Some events may have been missed due to a disconnected event stream.
        {#snippet actions()}
            <button
                class="cursor-pointer rounded-md border border-current/20 bg-white/10 px-2.5 py-1 text-xs transition-colors hover:bg-white/20"
                onclick={() => {
                    handleRefreshFromWatcher();
                    clearResumptionFailed();
                }}
            >
                Refresh
            </button>
        {/snippet}
    </Alert>

    <!-- Content: flex row with grid + side panel -->

    <div class="relative flex min-h-0 flex-1 gap-4">
        {#if error}
            <p class="text-error">Error: {error}</p>
        {:else}
            <!-- Image grid (full width on mobile, flex-1 on desktop).
                 DropUploadZone wraps the grid and handles drag/drop upload. -->
            <DropUploadZone
                {canUpload}
                blockedReason={uploadBlockedReason}
                {datasetName}
                ondrop={handleFilesDropped}
                onblockeddrop={(reason) => toast.warning(reason)}
            >
                <DatasetBrowser
                    class="mx-auto"
                    {datasetName}
                    items={images}
                    {isLoading}
                    {isLoadingMore}
                    selectedId={focusedItem?.id ?? null}
                    captioningId={captioningImageId}
                    onclick={handleItemClick}
                    onendreached={loadMore}
                />

                {#if !isLoading && images.length === 0}
                    <div class="empty-state">
                        <p class="text-lg">No images found</p>
                        <p class="mt-1 text-sm">This dataset may be empty or not yet scanned.</p>
                        {#if currentDataset?.source === 'upload'}
                            <button
                                class="btn-primary mt-4"
                                onclick={handleAddFilesClick}
                                type="button"
                            >
                                <SvgUpload class="h-4 w-4" />
                                Add Files
                            </button>
                            <p class="mt-2 text-xs text-muted">Or drop files onto the image grid</p>
                        {/if}
                    </div>
                {/if}
            </DropUploadZone>
        {/if}

        <SidePanel
            {datasetName}
            source={currentDataset?.source}
            {focusedItem}
            bind:activeTab={panelTab}
            bind:open={panelOpen}
            onpanelclose={handlePanelClose}
            onimagedelete={handleImageDelete}
        />
    </div>
</div>

<!-- Add Files dialog (mounted only when open). Pre-populated with dropped
     files, or empty when opened from the empty-state button. -->
{#if addFilesDialogOpen}
    <AddFilesDialog
        open={addFilesDialogOpen}
        {datasetName}
        files={droppedFiles}
        onclose={handleAddFilesClose}
        oncomplete={handleAddFilesComplete}
    />
{/if}
