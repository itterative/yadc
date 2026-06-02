<script lang="ts">
  import { page } from "$app/stores";
  import { onMount } from "svelte";
  import { browser } from "$app/environment";
  import DatasetBrowser from "$lib/components/dataset/DatasetBrowser.svelte";
  import SidePanel from "./SidePanel.svelte";
  import SvgChevronLeft from "$lib/icons/SvgChevronLeft.svelte";
  import SvgSpinner from "$lib/icons/SvgSpinner.svelte";
  import type { CaptionOptions } from "$lib/stores/captionOptions";
  import { captioningStatus, pendingDatasetChanges, clearPendingDatasetChange, registerJobId, resumptionFailed, clearResumptionFailed } from "$lib/stores/events";
  import { toast } from "$lib/stores/toasts";
  import { API_BASE } from "$lib/api";
  import {
    fetchDatasets,
    fetchImages,
    startCaptioning,
    captionSingleImage,
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
  let panelOpen = $state(false);

  // --- Captioning state ---

  // NOTE: Grid tiles are NOT updated live during captioning — the SSE events
  // (CaptioningStatusEvent) only carry aggregate counts (processed/total/errors),
  // not individual image IDs. A full reload happens when captioning finishes
  // (see handleCaptioningDone). For live per-tile updates, the backend would
  // need to emit per-image events (e.g. CaptionedImageEvent with image_id).

  // Captioning start error — shown inline below header. Also fires a persistent
  // toast so the error is visible even if the user navigates away.
  let captioningError: string | null = $state(null);
  let captionDoneFired = $state(false);

  // Derive captioning state from the global SSE store
  let isCaptioning = $derived(
    $captioningStatus?.dataset_name === datasetName &&
    ($captioningStatus.status === "running" || $captioningStatus.status === "stopping"),
  );

  let isStopping = $derived(
    $captioningStatus?.dataset_name === datasetName && $captioningStatus.status === "stopping",
  );

  let captionPct = $derived(
    $captioningStatus?.total > 0
      ? Math.round(($captioningStatus.processed / $captioningStatus.total) * 100)
      : 0,
  );

  async function handleStopCaptioning() {
    try {
      const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`,
        { method: "DELETE" },
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        toast.error(`Failed to stop captioning: ${body.error || res.status}`);
      }
    } catch {
      toast.error("Failed to stop captioning: request failed");
    }
  }

  // Register job_id from incoming status events so dataset_changed events
  // from our own captioning are suppressed
  $effect(() => {
    const s = $captioningStatus;
    if (s?.dataset_name === datasetName && s.job_id && (s.status === "running" || s.status === "stopping")) {
      registerJobId(s.job_id);
    }
  });

  // Detect when captioning finishes (transition from active → terminal)
  $effect(() => {
    if (!isCaptioning && !captionDoneFired) {
      captionDoneFired = true;
      handleCaptioningDone();
    }
    if (isCaptioning) {
      captionDoneFired = false;
    }
  });

  // Live caption options from CaptionSettings (updated reactively as settings change)
  let captionOptions: CaptionOptions = $state({});

  // --- Filesystem watcher state ---

  let hasPendingChanges = $state(false);

  $effect(() => {
    const name = datasetName;
    return pendingDatasetChanges.subscribe((set) => {
      hasPendingChanges = set.has(name);
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
    panelOpen = true;
  }

  function handlePanelClose() {
    focusedItem = null;
    panelTab = "caption";
    panelOpen = false;
  }

  function handleCaptionUpdated(imageId: number, _caption: string) {
    void _caption;
    images = images.map((img) => (img.id === imageId ? { ...img, has_caption: true } : img));
  }

  async function handleCaptionImage(imageId: number) {
    try {
      const result = await captionSingleImage(datasetName, imageId, captionOptions as Record<string, unknown>);
      registerJobId(result.job_id);
      if (result.caption) {
        images = images.map((img) => (img.id === imageId ? { ...img, has_caption: true } : img));
      }
      return result.caption;
    } catch (e) {
      throw e instanceof Error ? e : new Error("Failed to caption image");
    }
  }

  async function handleStartCaptioning(options: CaptionOptions) {
    captioningError = null;
    captionOptions = options;
    panelOpen = false;
    try {
      const info = await startCaptioning(datasetName, options as Record<string, unknown>);
      registerJobId(info.job_id);
    } catch (e) {
      captioningError = e instanceof Error ? e.message : "Failed to start captioning";
      toast.error(captioningError);
    }
  }

  function handleCaptioningDone() {
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

</script>

<svelte:head>
  <title>{datasetName} — yadc</title>
</svelte:head>

<div class="flex flex-col h-full gap-4">
  <!-- Header -->
  <div class="flex items-center gap-4 flex-shrink-0">
    <a href="#/" class="text-gray-400 hover:text-white transition-colors p-2 -ml-2">
      <SvgChevronLeft class="w-6 h-6" />
    </a>
    <div class="flex-1 min-w-0">
      <h1 class="text-xl font-bold text-white">{datasetName}</h1>
      <div class="min-h-[1.75rem] flex items-center mt-0.5 w-full">
      {#if isCaptioning}
        <div class="flex items-center gap-2 w-full">
          {#if isStopping}
            <SvgSpinner class="h-3.5 w-3.5 animate-spin text-yellow-400 flex-shrink-0" />
            <span class="text-sm text-yellow-300">Stopping…</span>
          {:else}
            <SvgSpinner class="h-3.5 w-3.5 animate-spin text-accent flex-shrink-0" />
            <span class="text-sm text-gray-400">
              Captioning… {$captioningStatus.processed}/{$captioningStatus.total}
            </span>
            <div class="flex-1 max-w-32 h-1.5 rounded-full bg-bg overflow-hidden">
              <div
                class="h-full rounded-full bg-accent transition-all duration-300 ease-out"
                style:width="{captionPct}%"
              ></div>
            </div>
            <span class="text-xs text-gray-500">{captionPct}%</span>
            <button
              class="px-2 py-0.5 text-xs rounded-md bg-error/20 border border-error/30 text-error hover:bg-error/30 transition-colors cursor-pointer disabled:opacity-50"
              onclick={handleStopCaptioning}
            >
              Stop
            </button>
          {/if}
        </div>
      {:else if currentDataset}
        <p class="text-sm text-gray-400">
          {currentDataset.image_count} images
          · {currentDataset.has_caption} captioned
          · {currentDataset.has_toml} with TOML
        </p>
      {/if}
      </div>
    </div>
  </div>

  <!-- Captioning error from start attempt -->
  {#if captioningError}
    <div class="alert-error">{captioningError}</div>
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

  <!-- SSE resumption failure notification -->
  {#if $resumptionFailed}
    <div class="rounded-lg bg-yellow-900/50 border border-yellow-700/50 px-4 py-2 flex items-center justify-between gap-4">
      <span class="text-sm text-yellow-200">Some events may have been missed due to a disconnected event stream.</span>
      <div class="flex items-center gap-2 flex-shrink-0">
        <button
          class="px-3 py-1.5 text-xs rounded-lg bg-yellow-600/40 border border-yellow-500/50 text-yellow-200 hover:bg-yellow-600/60 transition-colors cursor-pointer"
          onclick={() => { handleRefreshFromWatcher(); handleDismissResumptionFailed(); }}
        >
          Refresh
        </button>
        <button
          class="px-2 py-1.5 text-xs rounded-lg bg-transparent border border-yellow-500/30 text-yellow-300 hover:bg-yellow-600/20 transition-colors cursor-pointer"
          onclick={handleDismissResumptionFailed}
        >
          Dismiss
        </button>
      </div>
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
      bind:panelTab
      bind:open={panelOpen}
      bind:captionOptions
      onstartcaptioning={handleStartCaptioning}
      onpanelclose={handlePanelClose}
      oncaptionupdated={handleCaptionUpdated}
      oncaptionimage={handleCaptionImage}
    />
  </div>
</div>
