<script lang="ts">
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgPhoto from '$lib/icons/SvgPhoto.svelte';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SvgUpload from '$lib/icons/SvgUpload.svelte';
    import SvgWarning from '$lib/icons/SvgWarning.svelte';
    import {
        fetchDatasets,
        fetchImages,
        fetchCaption,
        thumbnailUrl,
        type DatasetInfo,
        type ImageInfo
    } from '$lib/stores/dataset';
    import { fetchImageAsDataUrl, readFileAsDataUrl } from '$lib/stores/prompts';
    import type { ExamplePair } from '$lib/stores/prompts';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, linkedController } from '$lib/abort';

    interface Props {
        examples: ExamplePair[];
        disabled?: boolean;
    }

    let { examples = $bindable([]), disabled = false }: Props = $props();

    const abort = createAbortContext();

    let pickerOpen = $state(false);
    let pickerError: string | null = $state(null);
    let isAddingFromPicker = $state(false);
    let datasets: DatasetInfo[] = $state([]);
    let selectedDataset: string = $state('');
    let datasetImages: ImageInfo[] = $state([]);
    let isLoadingImages = $state(false);

    let fileInput: HTMLInputElement | null = $state(null);
    let fileError: string | null = $state(null);

    $effect(() => {
        return () => abort.abort();
    });

    function addExample(ex: ExamplePair) {
        examples = [...examples, ex];
    }

    function removeExample(index: number) {
        examples = examples.filter((_, i) => i !== index);
    }

    function updateExample(index: number, patch: Partial<ExamplePair>) {
        examples = examples.map((ex, i) => (i === index ? { ...ex, ...patch } : ex));
    }

    async function pickFiles() {
        fileError = null;
        fileInput?.click();
    }

    async function handleFiles(event: Event) {
        const input = event.target as HTMLInputElement;
        const files = Array.from(input.files ?? []);
        if (files.length === 0) {
            return;
        }
        for (const file of files) {
            try {
                const { dataUrl, mime } = await readFileAsDataUrl(file);
                if (!mime.startsWith('image/')) {
                    fileError = `${file.name}: not an image file`;
                    continue;
                }
                const stem = file.name.replace(/\.[^.]+$/, '');
                addExample({ subject: stem, caption: '', image_data_url: dataUrl });
            } catch (e) {
                fileError = friendlyErrorMessage(e, `Failed to read ${file.name}`);
            }
        }
        // Reset so the same file can be re-selected.
        input.value = '';
    }

    async function openPicker() {
        pickerOpen = true;
        pickerError = null;
        datasetImages = [];
        selectedDataset = '';
        try {
            datasets = await fetchDatasets(abort.signal);
            if (datasets.length > 0) {
                selectedDataset = datasets[0].name;
                await loadDatasetImages();
            }
        } catch (e) {
            if (abort.signal.aborted) {
                return;
            }
            pickerError = friendlyErrorMessage(e, 'Failed to load datasets');
        }
    }

    function closePicker() {
        pickerOpen = false;
        datasetImages = [];
    }

    async function loadDatasetImages() {
        if (!selectedDataset) {
            return;
        }
        isLoadingImages = true;
        const controller = linkedController(abort.signal);
        try {
            const page = await fetchImages(selectedDataset, { limit: 24 }, controller.signal);
            if (controller.signal.aborted) {
                return;
            }
            datasetImages = page.images;
        } catch (e) {
            if (controller.signal.aborted) {
                return;
            }
            pickerError = friendlyErrorMessage(e, 'Failed to load images');
        } finally {
            if (!controller.signal.aborted) {
                isLoadingImages = false;
            }
        }
    }

    async function addImageFromDataset(image: ImageInfo) {
        if (!selectedDataset || isAddingFromPicker) {
            return;
        }
        isAddingFromPicker = true;
        pickerError = null;
        try {
            const { dataUrl } = await fetchImageAsDataUrl(selectedDataset, image.id);
            // Best-effort caption fetch — if the image has no caption
            // yet, the user fills it in manually below.
            let caption = '';
            try {
                const cap = await fetchCaption(selectedDataset, image.id);
                caption = cap.caption;
            } catch {
                /* no caption — leave empty for the user */
            }
            addExample({
                subject: image.file_name.replace(/\.[^.]+$/, ''),
                caption,
                image_data_url: dataUrl
            });
        } catch (e) {
            pickerError = friendlyErrorMessage(e, `Failed to add ${image.file_name}`);
        } finally {
            isAddingFromPicker = false;
        }
    }

    $effect(() => {
        if (pickerOpen && selectedDataset) {
            loadDatasetImages();
        }
    });
</script>

<input
    bind:this={fileInput}
    type="file"
    accept="image/*"
    multiple
    class="hidden"
    onchange={handleFiles}
/>

{#if examples.length === 0 && !pickerOpen}
    <div
        class="flex flex-col items-center gap-2 rounded-lg border border-dashed border-gray-600 bg-bg/30 px-4 py-6 text-center"
    >
        <SvgPhoto class="h-6 w-6 text-muted" />
        <p class="text-sm text-gray-400">No examples yet — add a few to improve style transfer.</p>
        <div class="flex gap-2">
            <button
                class="btn-secondary flex cursor-pointer items-center gap-1.5 text-sm"
                onclick={pickFiles}
                {disabled}
            >
                <SvgUpload class="h-3.5 w-3.5" />
                Add manual
            </button>
            <button
                class="btn-secondary flex cursor-pointer items-center gap-1.5 text-sm"
                onclick={openPicker}
                {disabled}
            >
                <SvgPlus class="h-3.5 w-3.5" />
                From dataset
            </button>
        </div>
        {#if fileError}
            <p class="text-xs text-error">{fileError}</p>
        {/if}
    </div>
{:else}
    <div class="space-y-2">
        <ul class="space-y-2">
            {#each examples as ex, i (i)}
                <li
                    class="flex gap-3 rounded-lg border border-border bg-bg/40 p-2"
                    data-testid="example-row"
                >
                    <!-- Thumbnail -->
                    <div
                        class="flex h-16 w-16 shrink-0 items-center justify-center overflow-hidden rounded-md bg-gray-800"
                    >
                        <img
                            src={ex.image_data_url}
                            alt={ex.subject}
                            class="h-full w-full object-cover"
                        />
                    </div>
                    <!-- Fields -->
                    <div class="flex min-w-0 flex-1 flex-col gap-1">
                        <input
                            class="input text-sm"
                            type="text"
                            placeholder="Subject"
                            value={ex.subject}
                            oninput={(e) =>
                                updateExample(i, { subject: (e.target as HTMLInputElement).value })}
                            {disabled}
                        />
                        <textarea
                            class="input h-14 resize-y text-sm"
                            placeholder="Caption"
                            value={ex.caption}
                            oninput={(e) =>
                                updateExample(i, {
                                    caption: (e.target as HTMLTextAreaElement).value
                                })}
                            {disabled}
                        ></textarea>
                    </div>
                    <!-- Remove -->
                    <button
                        class="self-start rounded-md p-1 text-gray-400 transition-colors hover:bg-bg hover:text-error"
                        onclick={() => removeExample(i)}
                        title="Remove example"
                        {disabled}
                    >
                        <SvgClose class="h-4 w-4" />
                    </button>
                </li>
            {/each}
        </ul>

        <div class="flex flex-wrap gap-2">
            <button
                class="btn-secondary flex cursor-pointer items-center gap-1.5 text-sm"
                onclick={pickFiles}
                {disabled}
            >
                <SvgUpload class="h-3.5 w-3.5" />
                Add manual
            </button>
            <button
                class="btn-secondary flex cursor-pointer items-center gap-1.5 text-sm"
                onclick={openPicker}
                {disabled}
            >
                <SvgPlus class="h-3.5 w-3.5" />
                From dataset
            </button>
        </div>

        {#if fileError}
            <p class="text-xs text-error">{fileError}</p>
        {/if}

        {#if pickerOpen}
            <div
                class="mt-2 rounded-lg border border-border bg-surface p-3"
                data-testid="dataset-picker"
            >
                <div class="mb-2 flex items-center justify-between">
                    <h4 class="text-sm font-medium text-gray-300">Pick from dataset</h4>
                    <button
                        class="rounded-md p-1 text-gray-400 transition-colors hover:bg-bg hover:text-white"
                        onclick={closePicker}
                        title="Close picker"
                    >
                        <SvgClose class="h-4 w-4" />
                    </button>
                </div>

                {#if datasets.length === 0}
                    <p class="text-sm text-gray-500">No datasets available.</p>
                {:else}
                    <div class="mb-2 flex gap-2">
                        <select
                            class="input cursor-pointer text-sm"
                            bind:value={selectedDataset}
                            disabled={isLoadingImages}
                        >
                            {#each datasets as ds (ds.name)}
                                <option value={ds.name}>{ds.name} ({ds.image_count})</option>
                            {/each}
                        </select>
                    </div>
                {/if}

                {#if isLoadingImages}
                    <p class="text-sm text-gray-500">Loading images…</p>
                {:else if datasetImages.length === 0 && selectedDataset}
                    <p class="text-sm text-gray-500">No images in this dataset.</p>
                {:else}
                    <div
                        class="grid max-h-80 grid-cols-[repeat(auto-fill,minmax(72px,1fr))] gap-1.5 overflow-y-auto"
                    >
                        {#each datasetImages as img (img.id)}
                            <button
                                class="group relative aspect-square cursor-pointer overflow-hidden rounded-md border border-border bg-gray-800 transition-opacity hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-50"
                                onclick={() => addImageFromDataset(img)}
                                disabled={isAddingFromPicker}
                                title={img.file_name}
                            >
                                <img
                                    src={thumbnailUrl(selectedDataset, img.id, 128)}
                                    alt={img.file_name}
                                    class="h-full w-full object-cover"
                                    loading="lazy"
                                />
                            </button>
                        {/each}
                    </div>
                {/if}

                {#if pickerError}
                    <p class="mt-2 flex items-center gap-1 text-xs text-error">
                        <SvgWarning class="h-3.5 w-3.5" />
                        {pickerError}
                    </p>
                {/if}
            </div>
        {/if}
    </div>
{/if}
