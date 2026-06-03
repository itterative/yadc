<script lang="ts">
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import { confirmDialog } from '$lib/stores/confirm';
    import { toast } from '$lib/stores/toasts';
    import { friendlyErrorMessage } from '$lib/api';
    import {
        fetchFolders,
        deleteDatasetItems,
        fetchDraftSummary,
        deleteDraftAll,
        type DatasetFolder,
        type DraftSummary
    } from '$lib/stores/dataset';

    interface Props {
        datasetName: string;
        source?: 'upload' | 'import' | 'create';
        onchanged?: () => void;
    }

    let { datasetName, source, onchanged }: Props = $props();

    let folders: DatasetFolder[] = $state([]);
    let foldersLoading = $state(false);
    let drafts: DraftSummary[] = $state([]);
    let draftsLoading = $state(false);

    const isUpload = $derived(source === 'upload');

    async function loadFolders() {
        foldersLoading = true;
        try {
            folders = await fetchFolders(datasetName);
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to load folders'));
        } finally {
            foldersLoading = false;
        }
    }

    async function loadDrafts() {
        draftsLoading = true;
        try {
            drafts = await fetchDraftSummary(datasetName);
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to load drafts'));
        } finally {
            draftsLoading = false;
        }
    }

    async function handleDeleteFolder(folder: DatasetFolder) {
        const ok = await confirmDialog.danger(
            `Delete "${folder.name}" and its ${folder.image_count} image${folder.image_count !== 1 ? 's' : ''}? This cannot be undone.`
        );
        if (!ok) {
            return;
        }
        try {
            await deleteDatasetItems(datasetName, [folder.path]);
            toast.success(`Deleted "${folder.name}"`);
            await loadFolders();
            onchanged?.();
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to delete folder'));
        }
    }

    async function handleDeleteDraft(draft: DraftSummary) {
        const ok = await confirmDialog.danger(
            `Delete the "${draft.name}" draft from ${draft.image_count} image${draft.image_count !== 1 ? 's' : ''}? This cannot be undone.`
        );
        if (!ok) {
            return;
        }
        try {
            const deleted = await deleteDraftAll(datasetName, draft.name);
            toast.success(
                `Deleted "${draft.name}" draft from ${deleted} image${deleted !== 1 ? 's' : ''}`
            );
            await loadDrafts();
            onchanged?.();
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to delete draft'));
        }
    }

    $effect(() => {
        const name = datasetName;
        if (!name) {
            return;
        }
        if (isUpload) {
            loadFolders();
        }
        loadDrafts();
    });
</script>

<div class="space-y-6 p-4">
    <!-- Folders section (upload datasets only) -->
    {#if isUpload}
        <section class="space-y-3">
            <h3 class="section-heading">Folders</h3>
            <p class="help-text">
                Folders are created automatically when uploading images. Deleting a folder removes
                all images inside it.
            </p>

            {#if foldersLoading}
                <SpinnerBlock class="py-6" size="h-5 w-5" label="Loading folders…" />
            {:else}
                {#each folders as folder (folder.path)}
                    <div class="card">
                        <div class="card-body flex items-center justify-between gap-3">
                            <div class="min-w-0 flex-1">
                                <p class="truncate font-mono text-sm text-gray-200">
                                    {folder.name}
                                </p>
                                <p class="text-xs text-gray-500">
                                    {folder.image_count} image{folder.image_count !== 1 ? 's' : ''}
                                </p>
                            </div>
                            {#if folder.can_delete}
                                <button
                                    class="cursor-pointer p-1 text-gray-500 transition-colors hover:text-error"
                                    onclick={() => handleDeleteFolder(folder)}
                                    title="Delete folder"
                                    aria-label="Delete folder"
                                >
                                    <SvgDelete class="h-4 w-4" />
                                </button>
                            {/if}
                        </div>
                    </div>
                {/each}
                {#if folders.length === 0}
                    <p class="text-center text-sm text-gray-500">No folders yet.</p>
                {/if}
            {/if}
        </section>
    {/if}

    <!-- Drafts section -->
    <section class="space-y-3">
        <h3 class="section-heading">Drafts</h3>
        <p class="help-text">
            Drafts are alternative captions stored alongside images. Deleting a draft removes it
            from all images that have it.
        </p>

        {#if draftsLoading}
            <SpinnerBlock class="py-6" size="h-5 w-5" label="Loading drafts…" />
        {:else}
            {#each drafts as draft (draft.name)}
                <div class="card">
                    <div class="card-body flex items-center justify-between gap-3">
                        <div class="min-w-0 flex-1">
                            <p class="truncate font-mono text-sm text-gray-200">
                                {draft.name}
                            </p>
                            <p class="text-xs text-gray-500">
                                {draft.image_count} image{draft.image_count !== 1 ? 's' : ''}
                            </p>
                        </div>
                        <button
                            class="cursor-pointer p-1 text-gray-500 transition-colors hover:text-error"
                            onclick={() => handleDeleteDraft(draft)}
                            title="Delete draft"
                            aria-label="Delete draft"
                        >
                            <SvgDelete class="h-4 w-4" />
                        </button>
                    </div>
                </div>
            {/each}
            {#if drafts.length === 0}
                <p class="text-center text-sm text-gray-500">No drafts yet.</p>
            {/if}
        {/if}
    </section>
</div>
