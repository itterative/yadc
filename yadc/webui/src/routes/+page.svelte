<script lang="ts">
    import { onMount } from 'svelte';
    import { browser } from '$app/environment';
    import {
        fetchDatasets,
        deleteDataset,
        thumbnailUrl,
        type DatasetInfo
    } from '$lib/stores/datasetImages';
    import { captioningStatus, type CaptioningStatus } from '$lib/stores/events';
    import Topbar from '$lib/components/ui/Topbar.svelte';
    import AddDatasetDialog from './AddDatasetDialog.svelte';
    import EditDatasetDialog from './EditDatasetDialog.svelte';
    import { confirmDialog } from '$lib/stores/confirm';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgPhoto from '$lib/icons/SvgPhoto.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { friendlyErrorMessage } from '$lib/api';

    let datasets: DatasetInfo[] = $state([]);
    let loading = $state(true);
    let error = $state('');
    let showAddDataset = $state(false);

    // Edit state
    let editingDataset: DatasetInfo | null = $state(null);

    onMount(() => {
        if (!browser) {
            return;
        }
        loadDatasets();
    });

    // Captioning status for a specific dataset (null if idle/not captioning)
    function captionStatusFor(name: string): CaptioningStatus | null {
        const s = $captioningStatus;
        if (s && s.dataset_name === name && s.status !== 'idle') {
            return s;
        }
        return null;
    }

    async function loadDatasets() {
        try {
            datasets = await fetchDatasets();
        } catch (e) {
            error = friendlyErrorMessage(e, 'Failed to load datasets');
        } finally {
            loading = false;
        }
    }

    function handleDatasetCreated(dataset: DatasetInfo) {
        datasets = [...datasets, dataset];
    }

    async function handleDelete(dataset: DatasetInfo) {
        const ok = await confirmDialog.danger(
            `Delete dataset "${dataset.name}"? This will unregister it and delete its state. Image files will not be removed.`
        );
        if (!ok) {
            return;
        }
        try {
            await deleteDataset(dataset.name);
            datasets = datasets.filter((d) => d.name !== dataset.name);
        } catch (e) {
            error = friendlyErrorMessage(e, 'Failed to delete dataset');
        }
    }
</script>

<Topbar>
    <div class="min-w-0 flex-1">
        <h1 class="text-xl font-bold text-white">Datasets</h1>

        <p class="mt-0.5 text-sm text-gray-400">
            {#if loading}
                ...
            {:else}
                {datasets.length} dataset{datasets.length === 1 ? '' : 's'}
            {/if}
        </p>
    </div>
</Topbar>

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
            <div class="card group relative">
                <a href="#/datasets/{dataset.name}" class="block hover:opacity-90">
                    {#if dataset.first_image_id != null}
                        <img
                            src={thumbnailUrl(dataset.name, dataset.first_image_id, 256)}
                            alt="{dataset.name} preview"
                            loading="lazy"
                            class="block aspect-video w-full bg-bg object-cover"
                        />
                    {:else}
                        <div class="flex aspect-video w-full items-center justify-center bg-border">
                            <SvgPhoto class="h-8 w-8 text-muted opacity-50" />
                        </div>
                    {/if}
                    <div class="card-body">
                        <div class="mb-1 flex items-center gap-2">
                            <h3 class="text-fg">{dataset.name}</h3>
                            {#if dataset.source === 'upload'}
                                <span class="badge-muted badge-sm">Managed</span>
                            {:else}
                                <span class="badge-muted badge-sm">External</span>
                            {/if}
                        </div>
                        {#if captionStatusFor(dataset.name)}
                            {@const cs = captionStatusFor(dataset.name)!}
                            <div class="mt-2 flex items-center gap-2 text-sm">
                                {#if cs.status === 'running'}
                                    <SvgSpinner class="h-3.5 w-3.5 animate-spin text-accent" />
                                    <span class="text-accent"
                                        >Captioning {cs.processed}/{cs.total}</span
                                    >
                                {:else if cs.status === 'stopping'}
                                    <SvgSpinner class="h-3.5 w-3.5 animate-spin text-yellow-400" />
                                    <span class="text-yellow-300">Stopping…</span>
                                {:else if cs.status === 'done'}
                                    <span class="text-success">✓ Complete</span>
                                {:else if cs.status === 'cancelled'}
                                    <span class="text-warning">⊘ Cancelled</span>
                                {:else if cs.status === 'error'}
                                    <span class="text-error">✗ Error</span>
                                {/if}
                            </div>
                        {:else if dataset.image_count > 0}
                            <div class="mt-2 flex items-center gap-1 text-sm text-muted">
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

                <!-- Action buttons (top-right corner) -->
                <div
                    class="absolute top-2 right-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 max-lg:opacity-100"
                >
                    <button
                        class="cursor-pointer rounded-md bg-black/60 p-1.5 text-gray-300 hover:bg-black/80 hover:text-white"
                        title="Edit dataset"
                        onclick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            editingDataset = dataset;
                        }}
                    >
                        <SvgEdit class="h-4 w-4" />
                    </button>
                    <button
                        class="cursor-pointer rounded-md bg-black/60 p-1.5 text-gray-300 hover:bg-black/80 hover:text-error"
                        title="Delete dataset"
                        onclick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            handleDelete(dataset);
                        }}
                    >
                        <SvgDelete class="h-4 w-4" />
                    </button>
                </div>
            </div>
        {/each}
        <button
            class="card flex min-h-56 cursor-pointer items-center justify-center border-dashed border-gray-600 hover:border-accent"
            onclick={() => (showAddDataset = true)}
            title="Add dataset"
        >
            <span class="text-sm text-gray-400">+ Add Dataset</span>
        </button>
    </div>

    <!-- Edit dialog -->
    {#if editingDataset}
        <EditDatasetDialog
            open={true}
            datasetName={editingDataset.name}
            source={editingDataset.source}
            onclose={() => (editingDataset = null)}
            onsaved={loadDatasets}
        />
    {/if}
{/if}

<AddDatasetDialog
    open={showAddDataset}
    onclose={() => (showAddDataset = false)}
    oncreated={handleDatasetCreated}
/>
