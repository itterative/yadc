<script lang="ts">
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgCheck from '$lib/icons/SvgCheck.svelte';
    import SvgCopy from '$lib/icons/SvgCopy.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgSparkle from '$lib/icons/SvgSparkle.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import Card from '$lib/components/ui/Card.svelte';
    import ActionBar from '$lib/components/ui/ActionBar.svelte';
    import ActionBarItem from '$lib/components/ui/ActionBarItem.svelte';
    import SvgHistory from '$lib/icons/SvgHistory.svelte';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import SvgUpgrade from '$lib/icons/SvgUpgrade.svelte';
    import ContextMenu from '$lib/components/ui/ContextMenu.svelte';
    import { confirmDialog } from '$lib/stores/confirm';
    import { captionOptions } from '$lib/stores/caption';
    import RefineDialog from './RefineDialog.svelte';
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import type { CaptionData, HistoryEntry, ImageInfo } from '$lib/stores/dataset';
    import { deleteRefineResult } from '$lib/stores/dataset/api';
    import { friendlyErrorMessage } from '$lib/api';
    import { PasswordPromptCancelled } from '$lib/stores/passwordPrompt';
    import { toast } from '$lib/stores/toasts';
    import { autosize } from '$lib/actions/autosize';

    interface Props {
        datasetName: string;
        item: ImageInfo;
        captionData: CaptionData | null;
        isLoadingCaption: boolean;
        captionError: string | null;
        isCaptioning: boolean;
        historyEntries: HistoryEntry[];
        isRestoring: boolean;
        copiedKey: string | null;
        onCaptioningStart: () => Promise<void>;
        onCaptioningCancel: () => Promise<void>;
        onSaveCaption: (text: string) => Promise<void>;
        onRestoreHistory: (index: number) => Promise<void>;
        onDeleteHistory: (hash: string) => Promise<void>;
        onDeleteDraft: (name: string) => Promise<void>;
        onPromoteDraft: (text: string) => Promise<void>;
        onWriteDraft: (name: string, text: string) => Promise<void>;
        onCopy: (key: string, text: string) => void;
    }

    let {
        datasetName,
        item,
        captionData,
        isLoadingCaption,
        captionError,
        isCaptioning,
        historyEntries,
        isRestoring,
        copiedKey,
        onCaptioningStart,
        onCaptioningCancel,
        onSaveCaption,
        onRestoreHistory,
        onDeleteHistory,
        onDeleteDraft,
        onPromoteDraft,
        onWriteDraft,
        onCopy
    }: Props = $props();

    let isEditing = $state(false);
    let editCaption = $state('');
    let isSavingCaption = $state(false);
    let isCancelling = $state(false);
    let captioningError: string | null = $state(null);
    let refineInitialCaption = $state<{
        source: 'caption' | 'draft';
        text: string;
        draftName?: string;
    } | null>(null);

    let activeDraftName = $derived($captionOptions?.draft?.trim() || '');

    // Reset local state when item changes
    $effect(() => {
        // Track item.id as a dependency
        void item.id;
        isEditing = false;
        captioningError = null;
    });

    // Sync editCaption from captionData when not editing
    $effect(() => {
        if (!isEditing) {
            editCaption = captionData?.caption || '';
        }
    });

    function handleStartEdit() {
        editCaption = captionData?.caption || '';
        isEditing = true;
    }

    function handleCancelEdit() {
        isEditing = false;
        editCaption = captionData?.caption || '';
    }

    async function handleSave() {
        if (!isEditing) {
            return;
        }
        isSavingCaption = true;
        try {
            await onSaveCaption(editCaption);
            isEditing = false;
        } catch {
            // Parent already set captionError — stay in edit mode so the user can retry
        } finally {
            isSavingCaption = false;
        }
    }

    async function handleCaptionImage() {
        captioningError = null;
        try {
            await onCaptioningStart();
        } catch (e) {
            if (e instanceof PasswordPromptCancelled) {
                return;
            }
            captioningError = friendlyErrorMessage(e, 'Failed to caption image');
        }
    }

    async function handleCancelSingleCaptioning() {
        isCancelling = true;
        try {
            await onCaptioningCancel();
        } finally {
            isCancelling = false;
        }
    }

    async function handleRestoreClick(index: number) {
        const ok = await confirmDialog.warning(
            'Restore this revision? The current caption and extras will be saved to history first.'
        );
        if (!ok) {
            return;
        }
        try {
            await onRestoreHistory(index);
        } catch {
            // Parent already set captionError
        }
    }

    async function handleDeleteHistoryClick(entryHash: string) {
        try {
            await onDeleteHistory(entryHash);
        } catch {
            // Parent already set captionError
        }
    }

    async function handleDeleteDraftClick(name: string) {
        try {
            await onDeleteDraft(name);
        } catch {
            // Parent already set captionError
        }
    }

    async function handlePromoteDraft(text: string) {
        try {
            await onPromoteDraft(text);
        } catch {
            // Parent already set captionError
        }
    }

    function escapeParentheses(text: string): string {
        return text.replace(/([()])/g, '\\$1');
    }

    async function copyText(text: string, escape = false) {
        if (!text) {
            return;
        }
        const toCopy = escape ? escapeParentheses(text) : text;
        try {
            await navigator.clipboard.writeText(toCopy);
            toast.success(escape ? 'Copied with escaped parentheses' : 'Copied to clipboard');
        } catch {
            toast.warning('Failed to copy to clipboard');
        }
    }

    function makeCopyItems(text: string) {
        return [
            { label: 'Copy', onClick: () => copyText(text) },
            { label: 'Copy with escape', onClick: () => copyText(text, true) }
        ];
    }
</script>

<div class="space-y-4">
    <!-- Caption -->
    <div>
        <h3 class="mb-2 text-sm font-medium text-gray-300">Caption</h3>

        {#if isLoadingCaption}
            <SpinnerBlock size="h-4 w-4" label="Loading caption..." />
        {:else if captionError}
            <p class="text-sm text-error">{captionError}</p>
        {:else}
            <Card class="relative">
                <!-- Copy button — anchored to the caption box, not the scrollable
                     text area, so it stays put while the user scrolls the text. -->
                {#if captionData && captionData.caption && !isEditing && !isCaptioning}
                    <ContextMenu
                        class="absolute top-2 right-2 z-10"
                        items={makeCopyItems(captionData.caption)}
                    >
                        <button
                            type="button"
                            class="cursor-pointer rounded p-1 text-gray-400 transition-colors hover:bg-gray-700 hover:text-gray-200"
                            aria-label="Copy caption"
                            title="Copy caption"
                            onclick={() => onCopy('caption', captionData!.caption!)}
                        >
                            {#if copiedKey === 'caption'}
                                <SvgCheck class="h-4 w-4 text-success" />
                            {:else}
                                <SvgCopy class="h-4 w-4" />
                            {/if}
                        </button>
                    </ContextMenu>
                {/if}

                <!-- Text area — only this part scrolls. Overlays are siblings
                     of the scroll container so they stay fixed over the visible
                     area instead of scrolling with the content. -->
                <div class="relative">
                    <div class="max-h-60 overflow-y-auto">
                        {#if isEditing}
                            <textarea
                                bind:value={editCaption}
                                class="w-full resize-none bg-transparent p-3 font-mono text-sm whitespace-pre-wrap text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
                                placeholder="Enter caption..."
                                use:autosize
                            ></textarea>
                        {:else if captionData}
                            {#if captionData.caption}
                                <pre
                                    class="p-3 pr-10 text-sm whitespace-pre-wrap text-gray-200">{captionData.caption}</pre>
                            {:else}
                                <pre
                                    class="p-3 text-sm whitespace-pre-wrap text-gray-500 italic">No caption</pre>
                            {/if}
                        {/if}
                    </div>

                    {#if isCaptioning}
                        <div
                            class="absolute inset-0 z-10 flex items-center justify-center gap-2 bg-black/60 text-sm text-gray-300"
                        >
                            <SvgSpinner class="h-4 w-4 animate-spin" />
                            <span
                                >{activeDraftName
                                    ? `Generating draft…`
                                    : 'Generating caption…'}</span
                            >
                        </div>
                    {:else if captioningError}
                        <div
                            class="absolute inset-x-0 top-0 z-10 bg-error/90 px-3 py-1.5 text-center text-sm text-white"
                        >
                            {captioningError}
                        </div>
                    {/if}
                </div>

                <!-- Action bar — footer of the caption box. -->
                <ActionBar>
                    {#if isEditing}
                        <ActionBarItem
                            onclick={handleCancelEdit}
                            disabled={isSavingCaption}
                            icon={SvgClose}
                            variant="secondary"
                        >
                            Cancel
                        </ActionBarItem>
                        <ActionBarItem
                            onclick={handleSave}
                            disabled={isSavingCaption}
                            icon={SvgCheck}
                            variant="primary"
                        >
                            {isSavingCaption ? 'Saving…' : 'Save'}
                        </ActionBarItem>
                    {:else if isCaptioning}
                        <ActionBarItem
                            onclick={handleCancelSingleCaptioning}
                            disabled={isCancelling}
                            icon={SvgClose}
                            variant="danger"
                        >
                            {isCancelling ? 'Cancelling…' : 'Cancel'}
                        </ActionBarItem>
                    {:else if captionData}
                        <ActionBarItem
                            onclick={handleCaptionImage}
                            icon={SvgSparkle}
                            variant="primary"
                        >
                            {activeDraftName ? `Draft (${activeDraftName})` : 'Caption'}
                        </ActionBarItem>
                        {#if captionData.caption}
                            <ActionBarItem
                                onclick={() => {
                                    refineInitialCaption = {
                                        source: 'caption',
                                        text: captionData?.caption || ''
                                    };
                                }}
                                disabled={isCaptioning}
                                icon={SvgRefresh}
                                variant="secondary"
                            >
                                Refine
                            </ActionBarItem>
                        {/if}
                        <ActionBarItem onclick={handleStartEdit} icon={SvgEdit} variant="secondary">
                            Edit
                        </ActionBarItem>
                    {/if}
                </ActionBar>
            </Card>
        {/if}
    </div>

    <!-- Drafts -->
    {#if captionData?.drafts && Object.keys(captionData.drafts).length > 0}
        <div>
            <h3 class="mb-2 text-sm font-medium text-gray-300">Drafts</h3>
            <p class="mb-2 text-xs text-gray-500">
                Alternative captions generated by the model. Use
                <code class="text-gray-400">drafts.&lt;name&gt;</code> in your template to reference them.
                "Promote" copies the draft text as the active caption.
            </p>
            <div class="space-y-3">
                {#each Object.entries(captionData.drafts) as [name, text] (name)}
                    {@const draftKey = `draft:${name}`}
                    <Card>
                        <div class="flex items-center justify-between px-3 pt-3 pb-1">
                            <span class="text-xs font-medium text-accent">{name}</span>
                            {#if text}
                                <ContextMenu items={makeCopyItems(text)}>
                                    <button
                                        type="button"
                                        class="cursor-pointer rounded p-1 text-gray-400 transition-colors hover:bg-gray-700 hover:text-gray-200"
                                        aria-label={`Copy draft "${name}"`}
                                        title={`Copy draft "${name}"`}
                                        onclick={() => onCopy(draftKey, text)}
                                    >
                                        {#if copiedKey === draftKey}
                                            <SvgCheck class="h-4 w-4 text-success" />
                                        {:else}
                                            <SvgCopy class="h-4 w-4" />
                                        {/if}
                                    </button>
                                </ContextMenu>
                            {/if}
                        </div>
                        <div class="px-3 pb-3">
                            <pre
                                class="max-h-40 overflow-y-auto font-mono text-sm whitespace-pre-wrap text-gray-200">{text}</pre>
                        </div>
                        <ActionBar>
                            <ActionBarItem
                                onclick={() => handlePromoteDraft(text)}
                                icon={SvgUpgrade}
                                variant="primary"
                            >
                                Promote
                            </ActionBarItem>
                            <ActionBarItem
                                onclick={() => {
                                    refineInitialCaption = {
                                        source: 'draft',
                                        text,
                                        draftName: name
                                    };
                                }}
                                disabled={isCaptioning}
                                icon={SvgRefresh}
                                variant="secondary"
                            >
                                Refine
                            </ActionBarItem>
                            <ActionBarItem
                                onclick={() => handleDeleteDraftClick(name)}
                                icon={SvgDelete}
                                variant="danger"
                            >
                                Delete
                            </ActionBarItem>
                        </ActionBar>
                    </Card>
                {/each}
            </div>
        </div>
    {/if}

    <!-- History -->
    {#if captionData && historyEntries.length > 0}
        <div>
            <h3 class="mb-2 text-sm font-medium text-gray-300">History</h3>
            <p class="mb-2 text-xs text-gray-500">
                Showing last {historyEntries.length} revision{historyEntries.length !== 1
                    ? 's'
                    : ''} of caption and extras changes. "Restore" replaces the current state with a previous
                version (saving the current state first).
            </p>
            <div class="space-y-3">
                {#each historyEntries as entry (entry.index)}
                    {@const hasExtras = Object.keys(entry.extras).length > 0}
                    <Card>
                        <div class="px-3 pt-3 pb-1">
                            <span class="text-xs font-medium text-gray-400"
                                >Revision #{entry.index}</span
                            >
                        </div>
                        <div class="px-3 pb-3">
                            <pre
                                class="max-h-32 overflow-y-auto font-mono text-sm whitespace-pre-wrap text-gray-200">{entry.caption ||
                                    '(empty)'}</pre>
                            {#if hasExtras}
                                <pre
                                    class="mt-2 max-h-32 overflow-y-auto rounded bg-gray-900 p-2 text-xs text-gray-400">{JSON.stringify(
                                        entry.extras,
                                        null,
                                        2
                                    )}</pre>
                            {/if}
                        </div>
                        <ActionBar>
                            <ActionBarItem
                                onclick={() => handleRestoreClick(entry.index)}
                                disabled={isRestoring}
                                icon={SvgHistory}
                                variant="primary"
                            >
                                {isRestoring ? 'Restoring…' : 'Restore'}
                            </ActionBarItem>
                            <ActionBarItem
                                onclick={() => handleDeleteHistoryClick(entry.hash)}
                                icon={SvgDelete}
                                variant="danger"
                            >
                                Delete
                            </ActionBarItem>
                        </ActionBar>
                    </Card>
                {/each}
            </div>
        </div>
    {/if}

    <RefineDialog
        open={refineInitialCaption !== null}
        onclose={() => {
            refineInitialCaption = null;
        }}
        {datasetName}
        {item}
        source={refineInitialCaption?.source || 'caption'}
        draftName={refineInitialCaption?.draftName}
        currentCaption={refineInitialCaption?.text || ''}
        onaccept={async (text: string) => {
            if (refineInitialCaption?.source === 'draft' && refineInitialCaption?.draftName) {
                await onWriteDraft(refineInitialCaption.draftName, text);
            } else {
                await onSaveCaption(text);
            }
            // Best-effort cleanup of the cached refine result on the server.
            // 404 (nothing cached) and 409 (a newer refine is cached) are
            // both non-errors and just no-op. We don't want a stale cache
            // entry to surface the same refinement the user just accepted
            // when they next open the dialog.
            const acceptedSource = refineInitialCaption?.source || 'caption';
            const acceptedDraftName = refineInitialCaption?.draftName || '';
            deleteRefineResult(datasetName, item.id, text, acceptedSource, acceptedDraftName).catch(
                () => {
                    // Ignore — see comment above.
                }
            );
        }}
    />
</div>
