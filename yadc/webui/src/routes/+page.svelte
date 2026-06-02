<script lang="ts">
  import { onMount } from "svelte";
  import { browser } from "$app/environment";
  import {
    fetchDatasets,
    thumbnailUrl,
    type DatasetInfo,
  } from "$lib/stores/datasetImages";
  import AddDatasetDialog from "$lib/components/AddDatasetDialog.svelte";

  let datasets: DatasetInfo[] = $state([]);
  let loading = $state(true);
  let error = $state("");
  let showAddDataset = $state(false);

  onMount(() => {
    if (!browser) return;
    loadDatasets();
  });

  async function loadDatasets() {
    try {
      datasets = await fetchDatasets();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load datasets";
    } finally {
      loading = false;
    }
  }

  function handleDatasetCreated(dataset: DatasetInfo) {
    datasets = [...datasets, dataset];
  }
</script>

<div class="flex items-center justify-between mb-6">
  <h1 class="text-xl font-bold">Datasets</h1>
  <button class="btn-primary" onclick={() => (showAddDataset = true)} title="Add dataset">
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
    </svg>
    Add Dataset
  </button>
</div>

{#if loading}
  <p class="text-muted">Loading datasets...</p>
{:else if error}
  <p class="text-error">Error: {error}</p>
{:else if datasets.length === 0}
  <div class="empty-state py-12">
    <h2>No datasets found</h2>
    <p>Import an existing TOML config or create a new dataset to get started.</p>
    <button class="btn-primary mt-4" onclick={() => (showAddDataset = true)}>
      Add Dataset
    </button>
  </div>
{:else}
  <div class="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-4">
    {#each datasets as dataset (dataset.name)}
      <a href="#/datasets/{dataset.name}" class="card hover:border-accent">
        {#if dataset.first_image_id != null}
          <img
            src={thumbnailUrl(dataset.name, dataset.first_image_id, 256)}
            alt="{dataset.name} preview"
            loading="lazy"
            class="block aspect-video w-full object-cover bg-bg"
          />
        {:else}
          <div class="flex items-center justify-center aspect-video w-full bg-border">
            <svg class="w-8 h-8 text-muted opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
            </svg>
          </div>
        {/if}
        <div class="card-body">
          <h3 class="text-fg mb-1">{dataset.name}</h3>
          {#if dataset.image_count > 0}
            <div class="flex items-center gap-1 mt-2 text-sm text-muted">
              <span>{dataset.image_count} images</span>
              <span class="dot-separator">·</span>
              <span>{dataset.has_caption} captioned</span>
              <span class="dot-separator">·</span>
              <span>{dataset.has_toml} with TOML</span>
            </div>
          {:else}
            <p class="text-muted">No images</p>
          {/if}
        </div>
      </a>
    {/each}
  </div>
{/if}

<AddDatasetDialog open={showAddDataset} onclose={() => (showAddDataset = false)} oncreated={handleDatasetCreated} />
