<script lang="ts">
    import { page } from '$app/stores';
    import { onMount, untrack } from 'svelte';
    import { browser } from '$app/environment';
    import { get } from 'svelte/store';
    import DatasetBrowser from '$lib/components/dataset/DatasetBrowser.svelte';
    import SidePanel from './SidePanel.svelte';
    import SvgChevronLeft from '$lib/icons/SvgChevronLeft.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
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
    import { toast } from '$lib/stores/toasts';
    import { friendlyErrorMessage } from '$lib/api';
    import {
        fetchDatasets,
        fetchImages,
        fetchCaptioningStatus,
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
        if (s.status !== 'error' && s.status !== 'done') {
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

    let hasPendingChanges = $state(false);

    $effect(() => {
        const name = datasetName;
        return pendingDatasetChanges.subscribe((set) => {
            hasPendingChanges = set.has(name);
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

    function handleDismissResumptionFailed() {
        clearResumptionFailed();
    }

    async function loadInitial(name: string) {
        nextToken = null;
        isLoading = true;
        error = null;

        try {
            const page = await fetchImages(name, { limit: PAGE_SIZE });
            images = page.images;
            nextToken = page.next_token;
            hasMore = page.next_token !== null;
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

        if (status.status === 'error') {
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

    function handleRefreshFromWatcher() {
        if (!browser || !datasetName) {
            return;
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
</Topbar>

<div class="flex h-full flex-col gap-4">
    <!-- Filesystem change notification -->
    {#if hasPendingChanges}
        <div
            class="flex items-center justify-between rounded-lg border border-blue-700/50 bg-blue-900/50 px-4 py-2"
        >
            <span class="text-sm text-blue-200">Dataset files have changed</span>
            <button
                class="cursor-pointer rounded-lg border border-blue-500/50 bg-blue-600/40 px-3 py-1.5 text-xs text-blue-200 transition-colors hover:bg-blue-600/60"
                onclick={handleRefreshFromWatcher}
            >
                Refresh
            </button>
        </div>
    {/if}

    <!-- SSE resumption failure notification -->
    {#if $resumptionFailed}
        <div
            class="flex items-center justify-between gap-4 rounded-lg border border-yellow-700/50 bg-yellow-900/50 px-4 py-2"
        >
            <span class="text-sm text-yellow-200"
                >Some events may have been missed due to a disconnected event stream.</span
            >
            <div class="flex shrink-0 items-center gap-2">
                <button
                    class="cursor-pointer rounded-lg border border-yellow-500/50 bg-yellow-600/40 px-3 py-1.5 text-xs text-yellow-200 transition-colors hover:bg-yellow-600/60"
                    onclick={() => {
                        handleRefreshFromWatcher();
                        handleDismissResumptionFailed();
                    }}
                >
                    Refresh
                </button>
                <button
                    class="cursor-pointer rounded-lg border border-yellow-500/30 bg-transparent px-2 py-1.5 text-xs text-yellow-300 transition-colors hover:bg-yellow-600/20"
                    onclick={handleDismissResumptionFailed}
                >
                    Dismiss
                </button>
            </div>
        </div>
    {/if}

    <!-- Content: flex row with grid + side panel -->

    <div class="relative flex min-h-0 flex-1 gap-4">
        {#if error}
            <p class="text-error">Error: {error}</p>
        {:else}
            <!-- Image grid (full width on mobile, flex-1 on desktop) -->
            <div class="min-w-0 flex-1 overflow-y-auto">
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
                    </div>
                {/if}
            </div>
        {/if}

        <SidePanel
            {datasetName}
            {focusedItem}
            bind:activeTab={panelTab}
            bind:open={panelOpen}
            onpanelclose={handlePanelClose}
            onconfigsaved={handleRefreshFromWatcher}
        />
    </div>
</div>
