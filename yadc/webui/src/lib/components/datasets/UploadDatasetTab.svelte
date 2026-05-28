<script lang="ts">
    import FileDropZone from '$lib/components/ui/FileDropZone.svelte';
    import { formatBytes } from '$lib/format';
    import { toast } from '$lib/stores/toasts';
    import { uploadDataset, type DatasetInfo } from '$lib/stores/datasetImages';
    import { friendlyErrorMessage } from '$lib/api';

    const UPLOAD_EXTENSIONS = [
        '.jpg',
        '.jpeg',
        '.png',
        '.gif',
        '.bmp',
        '.webp',
        '.tiff',
        '.tif',
        '.ico',
        '.txt',
        '.toml',
        '.history~',
        '.draft~'
    ];

    interface Props {
        oncreated: (dataset: DatasetInfo) => void;
        onclose: () => void;
    }

    let { oncreated, onclose }: Props = $props();

    let name = $state('');
    let selectedFiles: File[] = $state([]);
    let isSubmitting = $state(false);
    let error: string | null = $state(null);
    let uploadProgress: { loaded: number; total: number } | null = $state(null);
    let uploadAbortController: AbortController | null = $state(null);

    function handleCancelUpload() {
        if (uploadAbortController) {
            uploadAbortController.abort();
            uploadAbortController = null;
        }
    }

    async function handleUpload() {
        const trimmed = name.trim();

        if (!trimmed) {
            error = 'Dataset name is required';
            return;
        }
        if (selectedFiles.length === 0) {
            error = 'At least one file is required';
            return;
        }

        isSubmitting = true;
        error = null;
        uploadProgress = { loaded: 0, total: 0 };
        uploadAbortController = new AbortController();

        try {
            const { dataset, warnings } = await uploadDataset(
                trimmed,
                selectedFiles,
                (progress) => {
                    uploadProgress = progress;
                },
                uploadAbortController.signal
            );
            if (warnings.length > 0) {
                toast.warning(
                    `${warnings.length} file${warnings.length === 1 ? '' : 's'} skipped`,
                    { details: warnings.slice(0, 5) }
                );
            }
            oncreated(dataset);
            onclose();
        } catch (e) {
            if (e instanceof DOMException && e.name === 'AbortError') {
                error = 'Upload cancelled';
            } else {
                error = friendlyErrorMessage(e, 'Failed to upload dataset');
            }
        } finally {
            isSubmitting = false;
            uploadProgress = null;
            uploadAbortController = null;
        }
    }
</script>

<div class="space-y-4 py-4">
    <div>
        <label class="label" for="upload-name">Dataset Name</label>
        <input
            id="upload-name"
            type="text"
            bind:value={name}
            class="input"
            placeholder="my-dataset"
        />
    </div>

    <FileDropZone
        bind:files={selectedFiles}
        disabled={isSubmitting}
        allowedExtensions={UPLOAD_EXTENSIONS}
    >
        <p class="help-text">
            Images, .txt captions, .toml extras, and sidecars (.draft~ / .history~) are supported.
        </p>
    </FileDropZone>

    <div class="text-xs text-muted">
        <p>
            <strong>Note:</strong> Only <strong>one level</strong> of folder nesting is supported — subdirectories
            are ignored.
        </p>
        <p class="mt-1">
            Invalid files (corrupted images, bad TOML) and orphan sidecars without a matching image
            are skipped. You’ll be notified if anything is dropped.
        </p>
    </div>

    {#if error}
        <div class="alert-error">{error}</div>
    {/if}

    {#if uploadProgress && isSubmitting}
        <div class="space-y-1">
            <div class="h-2 w-full rounded-full bg-gray-700">
                <div
                    class="h-2 rounded-full bg-accent transition-all"
                    style="width: {uploadProgress.total > 0
                        ? Math.round((uploadProgress.loaded / uploadProgress.total) * 100)
                        : 0}%"
                ></div>
            </div>
            <div class="text-xs text-gray-400">
                {formatBytes(uploadProgress.loaded)} / {formatBytes(uploadProgress.total)}
            </div>
        </div>
    {/if}

    <div class="btn-bar">
        {#if isSubmitting}
            <button
                class="btn w-full border-error/40 bg-error/20 px-4 py-2 text-sm text-error hover:bg-error/30"
                onclick={handleCancelUpload}
                type="button"
            >
                Cancel Upload
            </button>
        {:else}
            <button
                class="btn-primary"
                onclick={handleUpload}
                disabled={isSubmitting}
                type="button"
            >
                Upload
            </button>
        {/if}
    </div>
</div>
