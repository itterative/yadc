<script lang="ts">
    import { createDataset, importDataset, type DatasetInfo } from '$lib/stores/datasetImages';
    import { friendlyErrorMessage } from '$lib/api';

    interface Props {
        oncreated: (dataset: DatasetInfo) => void;
        onclose: () => void;
    }

    let { oncreated, onclose }: Props = $props();

    let createMode: 'paths' | 'import' = $state('paths');

    let name = $state('');
    let tomlPath = $state('');
    let imagePaths = $state('');

    let isSubmitting = $state(false);
    let error: string | null = $state(null);

    async function handleSubmit() {
        const trimmed = name.trim();

        if (!trimmed) {
            error = 'Dataset name is required';
            return;
        }

        isSubmitting = true;
        error = null;

        try {
            let dataset: DatasetInfo;

            if (createMode === 'import') {
                const path = tomlPath.trim();
                if (!path) {
                    error = 'TOML file path is required';
                    isSubmitting = false;
                    return;
                }
                dataset = await importDataset(trimmed, path);
            } else {
                const paths = imagePaths
                    .split('\n')
                    .map((p) => p.trim())
                    .filter((p) => p.length > 0);
                if (paths.length === 0) {
                    error = 'At least one image directory path is required';
                    isSubmitting = false;
                    return;
                }
                dataset = await createDataset(trimmed, paths);
            }

            oncreated(dataset);
            onclose();
        } catch (e) {
            error = friendlyErrorMessage(
                e,
                createMode === 'import' ? 'Failed to import dataset' : 'Failed to create dataset'
            );
        } finally {
            isSubmitting = false;
        }
    }
</script>

<div class="space-y-4 py-4">
    <div>
        <label class="label" for="create-name">Dataset Name</label>
        <input
            id="create-name"
            type="text"
            bind:value={name}
            class="input"
            placeholder="my-dataset"
        />
    </div>

    <div class="flex justify-center gap-4 text-sm text-muted px-4">
        <label class="flex cursor-pointer items-center gap-2">
            <input type="radio" bind:group={createMode} value="paths" class="accent-accent" />
            <span>Add image paths</span>
        </label>
        <label class="flex cursor-pointer items-center gap-2">
            <input type="radio" bind:group={createMode} value="import" class="accent-accent" />
            <span>Import TOML config</span>
        </label>
    </div>

    {#if createMode === 'import'}
        <div>
            <label class="label" for="create-toml">TOML Config Path</label>
            <input
                id="create-toml"
                type="text"
                bind:value={tomlPath}
                class="input"
                placeholder="/path/to/dataset.toml"
            />
            <p class="text-muted text-xs mt-4">Absolute path to an existing yadc dataset config TOML file.</p>
        </div>
    {:else}
        <div>
            <label class="label" for="create-paths">Image Directories</label>
            <textarea
                id="create-paths"
                bind:value={imagePaths}
                rows={4}
                class="input resize-y"
                placeholder="/path/to/images&#10;/another/image/dir"
            ></textarea>
            <p class="text-muted text-xs mt-4">
                One directory path per line. Each directory will be scanned for images.
            </p>
        </div>
    {/if}

    {#if error}
        <div class="alert-error">{error}</div>
    {/if}

    <div class="btn-bar">
        <button class="btn-primary" onclick={handleSubmit} disabled={isSubmitting} type="button">
            {isSubmitting
                ? createMode === 'import'
                    ? 'Importing…'
                    : 'Creating…'
                : createMode === 'import'
                  ? 'Import'
                  : 'Create'}
        </button>
    </div>
</div>
