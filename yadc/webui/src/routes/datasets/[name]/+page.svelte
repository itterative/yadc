<script lang="ts">
  import { page } from "$app/stores";
  import { onMount } from "svelte";
  import { browser } from "$app/environment";
  import DatasetBrowser from "$lib/components/dataset/DatasetBrowser.svelte";
  import CaptionProgress from "$lib/components/dataset/CaptionProgress.svelte";
  import SidePanel from "./SidePanel.svelte";
  import type { CaptionOptions } from "$lib/stores/captionOptions";
  import { pendingDatasetChanges, clearPendingDatasetChange } from "$lib/stores/events";
  import {
    fetchDatasets,
    fetchImages,
    startCaptioning,
    type DatasetInfo,
    type ImageInfo,
  } from "$lib/stores/datasetImages";

  let datasetName = $derived($page.params.name ?? "");

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
  const MOBILE_BREAKPOINT = 1024;
  let windowWidth = $state(0);
  let isMobile = $derived(windowWidth > 0 && windowWidth < MOBILE_BREAKPOINT);
  let mobilePanelOpen = $state(false);

  // --- Captioning state ---

  // NOTE: Grid tiles are NOT updated live during captioning — the SSE events
  // (CaptioningStatusEvent) only carry aggregate counts (processed/total/errors),
  // not individual image IDs. A full reload happens when captioning finishes
  // (see handleCaptioningDone). For live per-tile updates, the backend would
  // need to emit per-image events (e.g. CaptionedImageEvent with image_id).

  let isCaptioning = $state(false);
  let captioningError: string | null = $state(null);

  // --- Filesystem watcher state ---

  let hasPendingChanges = $state(false);

  $effect(() => {
    const name = datasetName;
    return pendingDatasetChanges.subscribe((set) => {
      hasPendingChanges = set.has(name);
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
    } catch (e) {
      console.error("[dataset page] fetch failed", e);
      error = e instanceof Error ? e.message : "Failed to load images";
    } finally {
      isLoading = false;
    }
  }

  async function loadMore() {
    if (!datasetName) return;
    if (isLoadingMore || !hasMore || nextToken === null) return;
    isLoadingMore = true;

    try {
      const page = await fetchImages(datasetName, { limit: PAGE_SIZE, afterId: parseInt(nextToken) });
      images = [...images, ...page.images];
      nextToken = page.next_token;
      hasMore = page.next_token !== null;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load more images";
    } finally {
      isLoadingMore = false;
    }
  }

  // React to dataset name changes
  $effect(() => {
    const name = datasetName;
    if (!browser || !name) return;
    loadInitial(name);
  });

  // Find the dataset info from the list
  $effect(() => {
    if (datasets.length > 0 && datasetName) {
      currentDataset = datasets.find((d) => d.name === datasetName) ?? null;
    }
  });

  onMount(() => {
    if (!browser) return;
    (async () => {
      try {
        datasets = await fetchDatasets();
      } catch {
        datasets = [];
      }
    })();
  });

  // --- Side panel state ---

  let panelTab: "caption" | "details" = $state("caption");

  function handleItemClick(item: ImageInfo) {
    focusedItem = item;
    panelTab = "details";
    if (isMobile) mobilePanelOpen = true;
  }

  function handlePanelClose() {
    focusedItem = null;
    panelTab = "caption";
    if (isMobile) mobilePanelOpen = false;
  }

  function handleCaptionUpdated(imageId: number, _caption: string) {
    images = images.map((img) => (img.id === imageId ? { ...img, has_caption: true } : img));
  }

  async function handleStartCaptioning(options: CaptionOptions) {
    captioningError = null;
    try {
      await startCaptioning(datasetName, options as Record<string, unknown>);
      isCaptioning = true;
    } catch (e) {
      captioningError = e instanceof Error ? e.message : "Failed to start captioning";
    }
  }

  function handleCaptioningDone() {
    isCaptioning = false;
    // Refresh everything — caption counts and image states may have changed
    if (browser && datasetName) {
      loadInitial(datasetName);
      (async () => {
        try {
          datasets = await fetchDatasets();
        } catch { /* ignore */ }
      })();
    }
  }

  function handleRefreshFromWatcher() {
    if (!browser || !datasetName) return;
    clearPendingDatasetChange(datasetName);
    loadInitial(datasetName);
    (async () => {
      try {
        datasets = await fetchDatasets();
      } catch { /* ignore */ }
    })();
  }

  // Track window width for responsive behavior
  onMount(() => {
    if (!browser) return;
    windowWidth = window.innerWidth;
    const onResize = () => { windowWidth = window.innerWidth; };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  });
</script>

<svelte:head>
  <title>{datasetName} — yadc</title>
</svelte:head>

<div class="flex flex-col h-full gap-4">
  <!-- Header -->
  <div class="flex items-center gap-4 flex-shrink-0">
    <a href="#/" class="text-gray-400 hover:text-white transition-colors">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
      </svg>
    </a>
    <div>
      <h1 class="text-xl font-bold text-white">{datasetName}</h1>
      {#if currentDataset}
        <p class="text-sm text-gray-400">
          {currentDataset.image_count} images
          · {currentDataset.has_caption} captioned
          · {currentDataset.has_toml} with TOML
        </p>
      {/if}
    </div>
  </div>

  <!-- Captioning error from start attempt -->
  {#if captioningError}
    <div class="alert-error">{captioningError}</div>
  {/if}

  <!-- Captioning progress -->
  {#if isCaptioning}
    <CaptionProgress {datasetName} ondone={handleCaptioningDone} />
  {/if}

  <!-- Filesystem change notification -->
  {#if hasPendingChanges}
    <div class="rounded-lg bg-blue-900/50 border border-blue-700/50 px-4 py-2 flex items-center justify-between">
      <span class="text-sm text-blue-200">Dataset files have changed</span>
      <button
        class="px-3 py-1.5 text-xs rounded-lg bg-blue-600/40 border border-blue-500/50 text-blue-200 hover:bg-blue-600/60 transition-colors cursor-pointer"
        onclick={handleRefreshFromWatcher}
      >
        Refresh
      </button>
    </div>
  {/if}

  <!-- Content: flex row with grid + side panel -->
  {#if error}
    <p class="text-error">{error}</p>
  {/if}

  <div class="flex gap-4 flex-1 min-h-0 relative">
    <!-- Image grid (full width on mobile, flex-1 on desktop) -->
    <div class="flex-1 min-w-0 overflow-y-auto">
      <DatasetBrowser
        class="mx-auto"
        {datasetName}
        items={images}
        {isLoading}
        {isLoadingMore}
        selectedId={focusedItem?.id ?? null}
        onclick={handleItemClick}
        onendreached={loadMore}
      />

      {#if !isLoading && images.length === 0 && !error}
        <div class="empty-state">
          <p class="text-lg">No images found</p>
          <p class="text-sm mt-1">This dataset may be empty or not yet scanned.</p>
        </div>
      {/if}
    </div>

    <SidePanel
      {datasetName}
      focusedItem={focusedItem}
      {isMobile}
      bind:panelTab
      bind:open={mobilePanelOpen}
      onstartcaptioning={handleStartCaptioning}
      onpanelclose={handlePanelClose}
      oncaptionupdated={handleCaptionUpdated}
    />
  </div>
</div>
