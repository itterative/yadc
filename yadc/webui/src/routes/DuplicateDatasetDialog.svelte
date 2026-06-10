<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import { duplicateDataset, type DatasetInfo } from '$lib/stores/dataset';
    import { friendlyErrorMessage } from '$lib/api';

    interface Props {
        open: boolean;
        sourceDatasetName: string;
        onclose: () => void;
        onduplicated: (dataset: DatasetInfo) => void;
    }

    let { open, sourceDatasetName, onclose, onduplicated }: Props = $props();

    let newName = $state('');
    let isSubmitting = $state(false);
    let error: string | null = $state(null);

    // True after a 409 from the hardlink attempt; renders the
    // "retry as copy" warning and changes the button label.
    let hardlinkNotSupported = $state(false);

    // Reset on close
    $effect(() => {
        if (!open) {
            newName = '';
            isSubmitting = false;
            error = null;
            hardlinkNotSupported = false;
            return;
        }
        newName = `${sourceDatasetName}-copy`;
    });

    async function submit(mode: 'copy' | 'hardlink') {
        const trimmed = newName.trim();
        if (!trimmed) {
            error = 'New dataset name is required';
            return;
        }
        if (trimmed === sourceDatasetName) {
            error = 'New name must differ from the source name';
            return;
        }
        isSubmitting = true;
        error = null;
        try {
            const dataset = await duplicateDataset(sourceDatasetName, trimmed, mode);
            onduplicated(dataset);
            onclose();
        } catch (e) {
            // 409 with the hardlink mode means the filesystem
            // doesn't support hardlinks. Switch the dialog into
            // "retry as copy" mode; the user can confirm or cancel.
            const status = (e as Error & { status?: number }).status;
            if (status === 409 && mode === 'hardlink' && !hardlinkNotSupported) {
                hardlinkNotSupported = true;
            } else {
                error = friendlyErrorMessage(e, 'Failed to duplicate dataset');
            }
        } finally {
            isSubmitting = false;
        }
    }

    async function handlePrimary() {
        if (hardlinkNotSupported) {
            await submit('copy');
        } else {
            await submit('hardlink');
        }
    }

    function handleCancel() {
        onclose();
    }

    const primaryLabel = $derived(
        isSubmitting
            ? 'Duplicating…'
            : hardlinkNotSupported
              ? 'Proceed with full copy'
              : 'Duplicate'
    );
</script>

<Dialog class="dialog-panel max-w-lg" {open} {onclose}>
    <div class="p-5">
        <div class="dialog-header">
            <h2 class="dialog-title">Duplicate "{sourceDatasetName}"</h2>
            <button class="btn-close" onclick={onclose}>
                <SvgClose class="h-5 w-5" />
            </button>
        </div>

        <div class="space-y-4 py-4">
            <div>
                <label class="label" for="duplicate-name">New Dataset Name</label>
                <input
                    id="duplicate-name"
                    type="text"
                    bind:value={newName}
                    class="input"
                    placeholder="my-dataset-copy"
                />
            </div>

            {#if hardlinkNotSupported}
                <div class="alert-warning">
                    <p class="font-semibold">Hardlinks not supported on this filesystem</p>
                    <p class="mt-1 text-sm">
                        The duplicate must be a full copy, which uses about 2x the storage of the
                        original. Image bytes, captions, and metadata will all be independent from
                        the start.
                    </p>
                </div>
            {:else}
                <p class="text-sm text-muted">
                    Image bytes will be hardlinked (no extra storage); captions, history, and drafts
                    will be copied so you can edit them independently.
                </p>
            {/if}

            {#if error}
                <div class="alert-error">{error}</div>
            {/if}

            <div class="flex justify-end gap-2">
                <button
                    class="btn-secondary"
                    onclick={handleCancel}
                    disabled={isSubmitting}
                    type="button"
                >
                    Cancel
                </button>
                <button
                    class="btn-primary"
                    onclick={handlePrimary}
                    disabled={isSubmitting}
                    type="button"
                >
                    {primaryLabel}
                </button>
            </div>
        </div>
    </div>
</Dialog>
