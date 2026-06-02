<script lang="ts">
  import { page } from "$app/stores";
  import { onMount } from "svelte";
  import { browser } from "$app/environment";
  import DatasetBrowser from "$lib/components/DatasetBrowser.svelte";
  import ImageDetail from "$lib/components/ImageDetail.svelte";
  import {
    fetchDatasets,
    fetchImages,
    type DatasetInfo,
    type ImageInfo,
  } from "$lib/stores/datasetImages";

  let datasetName = $derived($page.params.name);

  let datasets: DatasetInfo[] = $state([]);
  let currentDataset: DatasetInfo | null = $state(null);
  let focusedItem: ImageInfo | null = $state(null);

  let images: ImageInfo[] = $state([]);
  let isLoading = $state(true);
  let isLoadingMore = $state(false);
  let hasMore = $state(true);
  let error: string | null = $state(null);

  let lastAfterId = 0;
  const PAGE_SIZE = 50;

  async function loadInitial(name: string) {
    lastAfterId = 0;
    isLoading = true;
    error = null;

    try {
      console.log("[dataset page] fetching images for", name);
      const page = await fetchImages(name, { limit: PAGE_SIZE, afterId: 0 });
      console.log("[dataset page] got", page.images.length, "images, next_token:", page.next_token);
      images = page.images;
      hasMore = page.next_token !== null;
      if (page.images.length > 0) {
        lastAfterId = page.images[page.images.length - 1].id;
      }
    } catch (e) {
      console.error("[dataset page] fetch failed", e);
      error = e instanceof Error ? e.message : "Failed to load images";
    } finally {
      isLoading = false;
      console.log("[dataset page] loading done, images:", images.length);
    }
  }

  async function loadMore() {
    if (isLoadingMore || !hasMore) return;
    isLoadingMore = true;

    try {
      const page = await fetchImages(datasetName, { limit: PAGE_SIZE, afterId: lastAfterId });
      images = [...images, ...page.images];
      hasMore = page.next_token !== null;
      if (page.images.length > 0) {
        lastAfterId = page.images[page.images.length - 1].id;
      }
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load more images";
    } finally {
      isLoadingMore = false;
    }
  }

  // React to dataset name changes
  $effect(() => {
    const name = datasetName;
    if (!name) return;
    console.log("[dataset page] loading images for", name);
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

  function handleItemClick(item: ImageInfo) {
    focusedItem = item;
  }

  function handleCaptionUpdated(imageId: number, _caption: string) {
    images = images.map((img) => (img.id === imageId ? { ...img, has_caption: true } : img));
  }
</script>

<svelte:head>
  <title>{datasetName} — yadc</title>
</svelte:head>

<div class="space-y-4">
  <!-- Header -->
  <div class="flex items-center gap-4">
    <a href="/" class="text-gray-400 hover:text-white transition-colors">
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

  <!-- Content -->
  {#if error}
    <p class="text-error">{error}</p>
  {/if}

  <DatasetBrowser
    class="container mx-auto"
    {datasetName}
    items={images}
    {isLoading}
    {isLoadingMore}
    onclick={handleItemClick}
    onendreached={loadMore}
  />

  {#if !isLoading && images.length === 0 && !error}
    <div class="text-center py-12 text-gray-400">
      <p class="text-lg">No images found</p>
      <p class="text-sm mt-1">This dataset may be empty or not yet scanned.</p>
    </div>
  {/if}
</div>

<ImageDetail
  {datasetName}
  item={focusedItem}
  onclose={() => (focusedItem = null)}
  oncaptionupdated={handleCaptionUpdated}
/>
