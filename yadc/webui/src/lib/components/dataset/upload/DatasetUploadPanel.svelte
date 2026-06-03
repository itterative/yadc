<script lang="ts">
    import { untrack } from 'svelte';
    import Alert from '$lib/components/ui/Alert.svelte';
    import FileDropZone from '$lib/components/ui/FileDropZone.svelte';
    import { formatBytes } from '$lib/format';
    import { toast } from '$lib/stores/toasts';
    import {
        uploadDataset,
        appendUploadDataset,
        commitStagingUpload,
        type DatasetInfo,
        type UploadProgressEvent,
        type UploadConflict
    } from '$lib/stores/dataset';
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

    type UploadPhase = 'uploading' | 'validating' | 'writing';

    interface Props {
        /** 'create' shows a name input; 'append' uses the provided datasetName. */
        mode: 'create' | 'append';
        /** Required when mode === 'append'. */
        datasetName?: string;
        /** One-time seed for the file list (e.g. from a drop event). Defaults to empty. */
        initialFiles?: File[];
        oncomplete: (dataset: DatasetInfo) => void;
        onclose: () => void;
    }

    let { mode, datasetName, initialFiles, oncomplete, onclose }: Props = $props();

    let name = $state('');
    // One-time seed from `initialFiles` (e.g. files dropped on the page before
    // the dialog opened). Re-seeding on prop changes is intentionally NOT
    // supported — the dialog is expected to be re-mounted for new file lists,
    // and updates after mount would be confusing. Use the FileDropZone inside
    // the panel to add more files after the dialog is open.
    let selectedFiles: File[] = $state(untrack(() => initialFiles ?? []));
    let isSubmitting = $state(false);
    let error: string | null = $state(null);
    let uploadAbortController: AbortController | null = $state(null);

    let uploadProgress: { loaded: number; total: number } | null = $state(null);
    let phase: UploadPhase | null = $state(null);
    let phaseProgress: { index: number; total: number; file: string } | null = $state(null);

    // Conflict resolution state
    let stagingId: string | null = $state(null);
    let conflicts: UploadConflict[] = $state([]);
    let conflictResolutions: Record<string, string> = $state({});
    let isCommitting = $state(false);

    function handleCancelUpload() {
        if (uploadAbortController) {
            uploadAbortController.abort();
            uploadAbortController = null;
        }
    }

    function setAllConflicts(action: string) {
        const next: Record<string, string> = {};
        for (const c of conflicts) {
            next[c.file] = action;
        }
        conflictResolutions = next;
    }

    async function handleCommit() {
        if (!stagingId || !datasetName) {
            return;
        }
        isCommitting = true;
        error = null;
        uploadAbortController = new AbortController();

        try {
            const { dataset, warnings } = await commitStagingUpload(
                datasetName,
                stagingId,
                conflictResolutions,
                undefined,
                uploadAbortController.signal
            );
            if (warnings.length > 0) {
                toast.warning(
                    `${warnings.length} file${warnings.length === 1 ? '' : 's'} skipped`,
                    { details: warnings.slice(0, 5) }
                );
            }
            oncomplete(dataset);
            resetForm();
            onclose();
        } catch (e) {
            if (e instanceof DOMException && e.name === 'AbortError') {
                error = 'Commit cancelled';
            } else {
                error = friendlyErrorMessage(e, 'Failed to commit upload');
            }
        } finally {
            isCommitting = false;
            uploadAbortController = null;
        }
    }

    function resetForm() {
        name = '';
        selectedFiles = [];
        error = null;
        stagingId = null;
        conflicts = [];
        conflictResolutions = {};
    }

    async function handleSubmit() {
        const targetName = mode === 'append' ? datasetName! : name.trim();

        if (mode === 'create' && !targetName) {
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
        phase = 'uploading';
        phaseProgress = null;
        stagingId = null;
        conflicts = [];
        conflictResolutions = {};
        uploadAbortController = new AbortController();

        try {
            const uploadFn = mode === 'create' ? uploadDataset : appendUploadDataset;
            const { dataset, warnings } = await uploadFn(
                targetName,
                selectedFiles,
                (progress) => {
                    uploadProgress = progress;
                },
                (event: UploadProgressEvent) => {
                    if (event.phase === 'validating' || event.phase === 'writing') {
                        phase = event.phase;
                        phaseProgress = {
                            index: event.index ?? 0,
                            total: event.total ?? 0,
                            file: event.file ?? ''
                        };
                    } else if (event.phase === 'conflicts') {
                        stagingId = event.staging_id ?? null;
                        conflicts = event.conflicts ?? [];
                        // Default all conflicts to 'overwrite'
                        const defaults: Record<string, string> = {};
                        for (const c of conflicts) {
                            defaults[c.file] = 'overwrite';
                        }
                        conflictResolutions = defaults;
                    }
                },
                uploadAbortController.signal
            );
            if (warnings.length > 0) {
                toast.warning(
                    `${warnings.length} file${warnings.length === 1 ? '' : 's'} skipped`,
                    { details: warnings.slice(0, 5) }
                );
            }
            oncomplete(dataset);
            resetForm();
            onclose();
        } catch (e) {
            if (e instanceof DOMException && e.name === 'AbortError') {
                error = 'Upload cancelled';
            } else {
                error = friendlyErrorMessage(e, 'Failed to upload');
            }
        } finally {
            isSubmitting = false;
            uploadProgress = null;
            phase = null;
            phaseProgress = null;
            uploadAbortController = null;
        }
    }

    const phaseLabel: Record<UploadPhase, string> = {
        uploading: 'Uploading',
        validating: 'Validating',
        writing: 'Writing'
    };

    const submitLabel = $derived(mode === 'create' ? 'Upload' : 'Add Files');
</script>

<div class="space-y-4 py-4">
    {#if mode === 'create'}
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
    {/if}

    {#if conflicts.length > 0}
        <Alert variant="warning" title="File conflicts detected" dismissable>
            <p class="mb-2 text-sm text-fg">
                {conflicts.length} file{conflicts.length === 1 ? '' : 's'} already exist{conflicts.length ===
                1
                    ? 's'
                    : ''} in this dataset. Choose how to handle each one.
            </p>
            {#snippet actions()}
                <button class="btn-sm" onclick={() => setAllConflicts('skip')} type="button">
                    Skip All
                </button>
                <button class="btn-sm" onclick={() => setAllConflicts('overwrite')} type="button">
                    Overwrite All
                </button>
            {/snippet}
        </Alert>

        <div class="max-h-48 overflow-y-auto rounded-md border border-gray-700">
            <table class="w-full text-left text-sm">
                <thead class="sticky top-0 bg-gray-800 text-xs text-gray-400 uppercase">
                    <tr>
                        <th class="px-3 py-2">File</th>
                        <th class="px-3 py-2 text-right">Existing</th>
                        <th class="px-3 py-2 text-right">New</th>
                        <th class="px-3 py-2">Action</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-gray-700">
                    {#each conflicts as c (c.file)}
                        <tr>
                            <td class="px-3 py-2 font-mono text-xs">{c.file}</td>
                            <td class="px-3 py-2 text-right text-gray-400"
                                >{formatBytes(c.existing_size)}</td
                            >
                            <td class="px-3 py-2 text-right text-gray-400"
                                >{formatBytes(c.new_size)}</td
                            >
                            <td class="px-3 py-2">
                                <select
                                    class="input py-1 text-xs"
                                    bind:value={conflictResolutions[c.file]}
                                >
                                    <option value="overwrite">Overwrite</option>
                                    <option value="skip">Skip</option>
                                    <option value="keep_both">Keep Both</option>
                                </select>
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>

        {#if error}
            <Alert variant="error" dismissable ondismiss={() => (error = null)}>{error}</Alert>
        {/if}

        <div class="btn-bar">
            {#if isCommitting}
                <button
                    class="btn w-full border-error/40 bg-error/20 px-4 py-2 text-sm text-error hover:bg-error/30"
                    onclick={handleCancelUpload}
                    type="button"
                >
                    Cancel
                </button>
            {:else}
                <button
                    class="btn-secondary"
                    onclick={() => {
                        conflicts = [];
                        stagingId = null;
                    }}
                    type="button"
                >
                    Back
                </button>
                <button class="btn-primary" onclick={handleCommit} type="button">Apply</button>
            {/if}
        </div>
    {:else}
        <FileDropZone
            bind:files={selectedFiles}
            disabled={isSubmitting}
            allowedExtensions={UPLOAD_EXTENSIONS}
        >
            <p class="help-text">
                Images, .txt captions, .toml extras, and sidecars (.draft~ / .history~) are
                supported.
            </p>
        </FileDropZone>

        <div class="text-xs text-muted">
            <p>
                <strong>Note:</strong> Only <strong>one level</strong> of folder nesting is supported
                — subdirectories are ignored.
            </p>
            <p class="mt-1">
                Invalid files (corrupted images, bad TOML) and orphan sidecars without a matching
                image are skipped. You'll be notified if anything is dropped.
            </p>
        </div>

        {#if error}
            <Alert variant="error" dismissable ondismiss={() => (error = null)}>{error}</Alert>
        {/if}

        {#if isSubmitting && phase}
            <div class="space-y-1">
                {#if phase === 'uploading' && uploadProgress}
                    <div class="h-2 w-full rounded-full bg-gray-700">
                        <div
                            class="h-2 rounded-full bg-accent transition-all"
                            style="width: {uploadProgress.total > 0
                                ? Math.round((uploadProgress.loaded / uploadProgress.total) * 100)
                                : 0}%"
                        ></div>
                    </div>
                    <div class="text-xs text-gray-400">
                        {phaseLabel[phase]}: {formatBytes(uploadProgress.loaded)} /
                        {formatBytes(uploadProgress.total)}
                    </div>
                {:else if phaseProgress && phaseProgress.total > 0}
                    {@const pct = Math.round((phaseProgress.index / phaseProgress.total) * 100)}
                    <div class="h-2 w-full rounded-full bg-gray-700">
                        <div
                            class="h-2 rounded-full bg-accent transition-all"
                            style="width: {pct}%"
                        ></div>
                    </div>
                    <div class="text-xs text-gray-400">
                        {phaseLabel[phase]}
                        {phaseProgress.index}/{phaseProgress.total}
                        — {phaseProgress.file}
                    </div>
                {:else}
                    <div class="text-xs text-gray-400">{phaseLabel[phase]}…</div>
                {/if}
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
                    onclick={handleSubmit}
                    disabled={isSubmitting}
                    type="button"
                >
                    {submitLabel}
                </button>
            {/if}
        </div>
    {/if}
</div>
