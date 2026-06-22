<script lang="ts">
    import { SvelteSet } from 'svelte/reactivity';
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgCheck from '$lib/icons/SvgCheck.svelte';
    import {
        fetchDatasets,
        fetchImages,
        fetchCaption,
        thumbnailUrl,
        type DatasetInfo,
        type ImageInfo
    } from '$lib/stores/dataset';
    import { fetchImageAsDataUrl } from '$lib/stores/prompts';
    import type { ExamplePair } from '$lib/stores/prompts';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, linkedController } from '$lib/abort';

    interface Props {
        open: boolean;
        onadd: (examples: ExamplePair[]) => void;
        onclose: () => void;
    }

    let { open, onadd, onclose }: Props = $props();

    // Component-level abort scope (auto-links to the parent's). Aborted
    // on unmount; per-open fetches use linked controllers so closing the
    // dialog cancels the in-flight dataset/image loads.
    const abort = createAbortContext();

    let datasets: DatasetInfo[] = $state([]);
    let selectedDataset = $state('');
    let datasetImages: ImageInfo[] = $state([]);
    let isLoadingImages = $state(false);
    let selected = new SvelteSet<number>();
    let isAdding = $state(false);
    let addProgress: string | null = $state(null);
    let error: string | null = $state(null);

    $effect(() => {
        return () => abort.abort();
    });

    // On open: reset state + load the dataset list, defaulting to the
    // first (which triggers the image-load effect below).
    $effect(() => {
        if (!open) {
            return;
        }
        error = null;
        selected.clear();
        addProgress = null;
        isAdding = false;
        datasetImages = [];

        const controller = linkedController(abort.signal);
        (async () => {
            try {
                const list = await fetchDatasets(controller.signal);
                if (controller.signal.aborted) {
                    return;
                }
                datasets = list;
                if (list.length > 0) {
                    selectedDataset = list[0].name;
                }
            } catch (e) {
                if (controller.signal.aborted) {
                    return;
                }
                error = friendlyErrorMessage(e, 'Failed to load datasets');
            }
        })();

        return () => controller.abort();
    });

    // Load images whenever the selected dataset changes (including the
    // initial pick on open). Clearing the selection here keeps stale ids
    // from a previous dataset from matching the new one (image ids are
    // per-dataset, so id 1 in dataset A ≠ id 1 in dataset B).
    $effect(() => {
        if (!open || !selectedDataset) {
            return;
        }
        selected.clear();
        error = null;

        const controller = linkedController(abort.signal);
        isLoadingImages = true;
        (async () => {
            try {
                const page = await fetchImages(selectedDataset, { limit: 48 }, controller.signal);
                if (controller.signal.aborted) {
                    return;
                }
                datasetImages = page.images;
            } catch (e) {
                if (controller.signal.aborted) {
                    return;
                }
                error = friendlyErrorMessage(e, 'Failed to load images');
            } finally {
                if (!controller.signal.aborted) {
                    isLoadingImages = false;
                }
            }
        })();

        return () => controller.abort();
    });

    function toggleSelected(id: number) {
        if (selected.has(id)) {
            selected.delete(id);
        } else {
            selected.add(id);
        }
    }

    async function handleAdd() {
        if (selected.size === 0 || isAdding || !selectedDataset) {
            return;
        }
        isAdding = true;
        error = null;
        const picked = datasetImages.filter((img) => selected.has(img.id));
        const out: ExamplePair[] = [];
        try {
            // Sequential: fetchImageAsDataUrl pulls full-resolution media
            // (a few-shot example needs the real image, not a thumbnail),
            // so we avoid hammering the server with N concurrent media
            // fetches. The caption fetch is best-effort (no caption → the
            // user fills it in via the example's Edit action afterward).
            for (let i = 0; i < picked.length; i++) {
                const image = picked[i];
                addProgress = `Adding ${i + 1} of ${picked.length}…`;
                const { dataUrl } = await fetchImageAsDataUrl(
                    selectedDataset,
                    image.id,
                    abort.signal
                );
                let caption = '';
                try {
                    const cap = await fetchCaption(selectedDataset, image.id, abort.signal);
                    caption = cap.caption;
                } catch {
                    /* no caption — leave empty for the user to fill */
                }
                out.push({
                    subject: image.file_name.replace(/\.[^.]+$/, ''),
                    caption,
                    image_data_url: dataUrl
                });
            }
            onadd(out);
        } catch (e) {
            if (abort.signal.aborted) {
                return;
            }
            error = friendlyErrorMessage(e, 'Failed to add images');
        } finally {
            isAdding = false;
            addProgress = null;
        }
    }
</script>

<Dialog class="dialog-panel flex max-w-2xl flex-col overflow-hidden" {open} {onclose}>
    <div class="flex flex-col p-5">
        <div class="dialog-header">
            <h2 class="dialog-title">Add from dataset</h2>
            <button class="btn-close" onclick={onclose} aria-label="Close">
                <SvgClose class="h-5 w-5" />
            </button>
        </div>

        {#if error}
            <div class="alert-error mb-4">{error}</div>
        {/if}

        {#if datasets.length === 0 && !error}
            <p class="py-8 text-center text-sm text-gray-500">No datasets available.</p>
        {:else if datasets.length > 0}
            <select
                class="input mb-3 cursor-pointer text-sm"
                bind:value={selectedDataset}
                disabled={isAdding}
            >
                {#each datasets as ds (ds.name)}
                    <option value={ds.name}>{ds.name} ({ds.image_count})</option>
                {/each}
            </select>

            <div class="max-h-[55vh] overflow-y-auto">
                {#if isLoadingImages}
                    <p class="py-8 text-center text-sm text-gray-500">Loading images…</p>
                {:else if datasetImages.length === 0}
                    <p class="py-8 text-center text-sm text-gray-500">No images in this dataset.</p>
                {:else}
                    <div class="grid grid-cols-[repeat(auto-fill,minmax(80px,1fr))] gap-1.5">
                        {#each datasetImages as img (img.id)}
                            {@const isSelected = selected.has(img.id)}
                            <button
                                type="button"
                                class="relative aspect-square cursor-pointer overflow-hidden rounded-md border transition-colors disabled:cursor-not-allowed {isSelected
                                    ? 'border-accent'
                                    : 'border-border hover:border-gray-500'}"
                                onclick={() => toggleSelected(img.id)}
                                disabled={isAdding}
                                title={img.file_name}
                            >
                                <img
                                    src={thumbnailUrl(selectedDataset, img.id, 128)}
                                    alt={img.file_name}
                                    class="h-full w-full object-cover"
                                    loading="lazy"
                                />
                                {#if isSelected}
                                    <span
                                        class="absolute top-1 right-1 flex h-5 w-5 items-center justify-center rounded-full bg-accent text-black"
                                    >
                                        <SvgCheck class="h-3.5 w-3.5" />
                                    </span>
                                {/if}
                            </button>
                        {/each}
                    </div>
                {/if}
            </div>

            {#if addProgress}
                <p class="mt-2 text-xs text-accent">{addProgress}</p>
            {/if}

            <div class="btn-bar mt-4">
                <span class="mr-auto self-center text-xs text-gray-400">
                    {selected.size} selected
                </span>
                <button class="btn-secondary" onclick={onclose} disabled={isAdding}>Cancel</button>
                <button
                    class="btn-primary"
                    onclick={handleAdd}
                    disabled={selected.size === 0 || isAdding}
                >
                    {selected.size > 0 ? `Add ${selected.size}` : 'Add'}
                </button>
            </div>
        {/if}
    </div>
</Dialog>
