<script lang="ts">
    import { untrack } from 'svelte';
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import Card from '$lib/components/ui/Card.svelte';
    import ActionBar from '$lib/components/ui/ActionBar.svelte';
    import ActionBarItem from '$lib/components/ui/ActionBarItem.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import SvgCopy from '$lib/icons/SvgCopy.svelte';
    import SvgCheck from '$lib/icons/SvgCheck.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import ContextMenu from '$lib/components/ui/ContextMenu.svelte';
    import { autosize } from '$lib/actions/autosize';
    import { currentlyCaptioning, imageRefined, consumeImageRefined } from '$lib/stores/caption';
    import {
        refineCaption as refineCaptionAction,
        stopCaptioning as stopCaptioningAction
    } from '$lib/stores/caption';
    import { friendlyErrorMessage } from '$lib/api';
    import { PasswordPromptCancelled } from '$lib/stores/passwordPrompt';
    import { toast } from '$lib/stores/toasts';
    import type { ImageInfo } from '$lib/stores/dataset';
    import { fetchRefineResult } from '$lib/stores/dataset/api';

    interface Props {
        open: boolean;
        onclose: () => void;
        datasetName: string;
        item: ImageInfo;
        source: 'caption' | 'draft';
        draftName?: string;
        currentCaption: string;
        onaccept: (caption: string) => Promise<void>;
    }

    let { open, onclose, datasetName, item, source, draftName, currentCaption, onaccept }: Props =
        $props();

    // --- Non-reactive refs ---
    // The AbortController for the in-flight refine HTTP request. Not a
    // Svelte $state because nothing in the template depends on it; it's
    // just a handle the cancel button can pull.
    let refineController: AbortController | null = null;

    // --- Local state ---
    let caption: string = $state('');
    let feedback: string = $state('');
    let isRefining: boolean = $state(false);
    let isAccepting: boolean = $state(false);
    let isRestoringResult: boolean = $state(false);
    let refinedCaption: string | null = $state(null);
    let error: string | null = $state(null);
    let copiedKey: string | null = $state(null);
    let copyTimeout: ReturnType<typeof setTimeout> | null = $state(null);
    let isEditingCaption: boolean = $state(false);
    let isEditingRefined: boolean = $state(false);

    let wasCaptioning = $state(false);

    let thumbnailSrc = $derived(
        `/api/datasets/${encodeURIComponent(datasetName)}/images/${item.id}/thumbnail?size=768`
    );

    // Reset state when dialog opens; try to restore a server-side result.
    // Also consume any stale SSE refine event for this image so an old result
    // can't short-circuit a new refine request.
    //
    // ``untrack`` keeps the prop reads (currentCaption, source, draftName, ...)
    // from becoming effect dependencies. Otherwise a parent re-render that
    // passes a new prop value (e.g. refineInitialCaption replacing the same
    // text) would re-run this effect mid-session, clobbering the user's
    // in-progress feedback and wiping refinedCaption just as a result arrives.
    $effect(() => {
        if (!open) {
            return;
        }
        untrack(() => {
            caption = currentCaption;
            feedback = '';
            refinedCaption = null;
            error = null;
            isRefining = false;
            isAccepting = false;
            isRestoringResult = false;
            wasCaptioning = false;
            copiedKey = null;
            isEditingCaption = false;
            isEditingRefined = false;
            if (copyTimeout !== null) {
                clearTimeout(copyTimeout);
                copyTimeout = null;
            }
            // Discard any leftover controller from a previous open. The
            // old fetch was already rejected (if it was in-flight) when
            // we closed the dialog, so this is just a safety net.
            if (refineController !== null) {
                refineController.abort();
                refineController = null;
            }

            consumeImageRefined(item.id, source, draftName);

            // Check if the server already has a refine result for this image
            isRestoringResult = true;
            fetchRefineResult(datasetName, item.id, source, draftName)
                .then((result) => {
                    if (result !== null) {
                        refinedCaption = result;
                    }
                })
                .catch(() => {
                    // Ignore 404s and other errors silently
                })
                .finally(() => {
                    isRestoringResult = false;
                });
        });
    });

    // Single effect watches both stores to detect refine completion.
    //
    // - On success: imageRefined is set and currentlyCaptioning is cleared
    //   synchronously in the same SSE handler — we read the refined caption
    //   directly from the event.
    // - On error: imageRefined stays null, but currentlyCaptioning is cleared
    //   by the image_caption_error handler — the transition (wasCaptioning &&
    //   !current) catches this and stops the spinner.
    //
    // Both reads happen in one effect run, so there is no ordering race between
    // the two stores. The imageRefined check takes priority.
    $effect(() => {
        const refinedEvent = $imageRefined;
        const cc = $currentlyCaptioning;

        if (!isRefining) {
            return;
        }

        // Primary path: check for a refine result matching this image + source.
        if (
            refinedEvent &&
            refinedEvent.image_id === item.id &&
            refinedEvent.source === source &&
            (source !== 'draft' || refinedEvent.draft_name === draftName)
        ) {
            refinedCaption = refinedEvent.caption;
            isRefining = false;
            consumeImageRefined(item.id, source, draftName);
            wasCaptioning = false;
            return;
        }

        // Fallback: detect job completion without a refine result (error/cancellation).
        const captioningThisImage =
            cc !== null &&
            [...cc].some((t) => t.dataset_name === datasetName && t.image_id === item.id);
        if (wasCaptioning && !captioningThisImage) {
            isRefining = false;
        }
        wasCaptioning = captioningThisImage;
    });

    async function handleRefine(e?: SubmitEvent) {
        e?.preventDefault();
        if (!feedback.trim() || !caption.trim()) {
            return;
        }
        error = null;
        refinedCaption = null;
        isEditingCaption = false;
        isEditingRefined = false;
        isRefining = true;

        const controller = new AbortController();
        refineController = controller;
        try {
            await refineCaptionAction(
                datasetName,
                item.id,
                feedback.trim(),
                caption.trim(),
                source,
                draftName ?? '',
                controller.signal
            );
        } catch (e) {
            if (e instanceof PasswordPromptCancelled) {
                isRefining = false;
                refineController = null;
                return;
            }
            if (e instanceof DOMException && e.name === 'AbortError') {
                // Aborted either by handleCancelRefine (in-flight cancel) or
                // by the open-reset effect (user closed the dialog). Either
                // way, the controller is going away — just clear it.
                refineController = null;
                return;
            }
            error = friendlyErrorMessage(e, 'Failed to refine caption');
            isRefining = false;
            refineController = null;
        }
    }

    async function handleCancelRefine() {
        // Abort the in-flight HTTP request, then ask the server to stop
        // the job. The abort will reject the fetch; the server-side stop
        // ensures the job doesn't keep running unattended.
        if (refineController !== null) {
            refineController.abort();
            refineController = null;
        }
        // Best-effort — don't surface a toast for this stop (the user
        // already initiated the cancel from this dialog).
        void stopCaptioningAction(datasetName);
        isRefining = false;
    }

    async function handleAccept() {
        if (refinedCaption === null) {
            return;
        }
        isAccepting = true;
        try {
            await onaccept(refinedCaption);
            onclose();
        } catch {
            // Parent handles error
        } finally {
            isAccepting = false;
        }
    }

    function handleEditCaption() {
        isEditingCaption = true;
    }

    function handleCancelEditCaption() {
        caption = currentCaption;
        isEditingCaption = false;
    }

    function handleUpdateCaption() {
        // Local-only commit: exit edit mode and keep the current value.
        // The local caption is never synced back to the parent.
        isEditingCaption = false;
    }

    function handleEditRefined() {
        if (refinedCaption === null) {
            return;
        }
        isEditingRefined = true;
    }

    function handleUpdateRefined() {
        isEditingRefined = false;
    }

    function escapeParentheses(text: string): string {
        return text.replace(/([()])/g, '\\$1');
    }

    async function handleCopy(text: string, escape = false) {
        if (!text) {
            return;
        }
        const toCopy = escape ? escapeParentheses(text) : text;
        try {
            await navigator.clipboard.writeText(toCopy);
            toast.success(escape ? 'Copied with escaped parentheses' : 'Copied to clipboard');
            copiedKey = escape ? 'escaped' : 'plain';
            if (copyTimeout !== null) {
                clearTimeout(copyTimeout);
            }
            copyTimeout = setTimeout(() => {
                copiedKey = null;
            }, 1500);
        } catch {
            toast.warning('Failed to copy to clipboard');
        }
    }

    function makeCopyItems(text: string) {
        return [
            { label: 'Copy', onClick: () => handleCopy(text) },
            { label: 'Copy with escape', onClick: () => handleCopy(text, true) }
        ];
    }
</script>

<Dialog class="dialog-panel max-h-[85dvh] max-w-2xl overflow-y-auto" {open} {onclose}>
    <form class="p-5" onsubmit={handleRefine}>
        <!-- Header -->
        <div class="dialog-header">
            <div>
                <h2 class="dialog-title">Refine</h2>
                <p class="mt-0.5 text-xs text-gray-500">
                    {source === 'draft' ? (draftName ?? '') : 'caption'}
                </p>
            </div>
            <button type="button" class="btn-close" onclick={onclose} aria-label="Close">
                <SvgClose class="h-5 w-5" />
            </button>
        </div>

        <img
            src={thumbnailSrc}
            alt={item.file_name}
            class="-mx-5 mb-4 box-content h-32 w-[calc(100%+var(--spacing)*10)] max-w-none bg-gray-700 object-cover"
        />

        <!-- Card 1: Caption / draft to refine -->
        <div class="mb-4">
            <h3 class="mb-2 text-sm font-medium text-gray-300">
                {source === 'draft' ? 'Draft to refine' : 'Caption to refine'}
            </h3>
            <Card>
                <div class="max-h-60 overflow-y-auto">
                    {#if isEditingCaption}
                        <!-- svelte-ignore a11y_autofocus -->
                        <!-- Edit mode is user-initiated; focusing the input avoids an extra click. -->
                        <textarea
                            id="refine-caption"
                            bind:value={caption}
                            class="w-full resize-none bg-transparent p-3 font-mono text-sm whitespace-pre-wrap text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
                            placeholder={source === 'draft'
                                ? 'Draft text...'
                                : 'Current caption...'}
                            disabled={isRefining || isRestoringResult}
                            autofocus
                            use:autosize
                        ></textarea>
                    {:else}
                        <pre
                            class="p-3 font-mono text-sm whitespace-pre-wrap text-gray-200">{caption ||
                                (source === 'draft' ? 'No draft' : 'No caption')}</pre>
                    {/if}
                </div>
                <ActionBar>
                    {#if isEditingCaption}
                        <ActionBarItem
                            onclick={handleCancelEditCaption}
                            disabled={isRefining || isRestoringResult}
                            icon={SvgClose}
                            variant="secondary"
                        >
                            Cancel
                        </ActionBarItem>
                        <ActionBarItem
                            onclick={handleUpdateCaption}
                            disabled={isRefining || isRestoringResult}
                            icon={SvgCheck}
                            variant="primary"
                        >
                            Update
                        </ActionBarItem>
                    {:else}
                        <ActionBarItem
                            onclick={handleEditCaption}
                            disabled={isRefining || isRestoringResult}
                            icon={SvgEdit}
                        >
                            Edit
                        </ActionBarItem>
                    {/if}
                </ActionBar>
            </Card>
            <p class="help-text">Edit to tweak the context sent to the model.</p>
        </div>

        <!-- Card 2: Feedback + refined result -->
        <div class="mb-4">
            <h3 class="mb-2 text-sm font-medium text-gray-300">Feedback</h3>
            <Card class="text-gray-300">
                <!-- Inner wrapper gives the refining overlay a positioning
                     context so it covers the feedback + result but not the
                     ActionBar below. Mirrors the pattern used in the caption
                     subtab of ImageDetail. -->
                <div class="relative">
                    <!-- Feedback (always editable). -->
                    <!-- svelte-ignore a11y_autofocus -->
                    <!-- The dialog opens on user action; focusing the primary input avoids an extra click. -->
                    <textarea
                        id="refine-feedback"
                        bind:value={feedback}
                        class="min-h-[72px] w-full resize-none bg-transparent p-3 text-sm focus:ring-2 focus:ring-accent focus:outline-none"
                        placeholder="e.g. Make it shorter, add detail about the background..."
                        disabled={isRefining || isRestoringResult}
                        autofocus
                        use:autosize={{ maxHeight: 200 }}
                    ></textarea>

                    {#if isRestoringResult}
                        <div
                            class="flex items-center gap-2 border-t border-border p-3 text-sm text-gray-400"
                        >
                            <SvgSpinner class="h-4 w-4 animate-spin" />
                            <span>Restoring previous result…</span>
                        </div>
                    {:else if refinedCaption !== null}
                        <!-- Refined result, separated from the feedback by a top border. -->
                        <div class="border-t border-border">
                            <p class="px-3 pt-2 text-xs text-gray-500">Refined result</p>
                            <div class="relative">
                                <!-- Copy button anchored to the result so it stays put
                                     while the user scrolls the text. Right-click for
                                     the "Copy with escape" variant — same pattern as
                                     the caption subtab of ImageDetail. -->
                                {#if !isEditingRefined}
                                    <ContextMenu
                                        class="absolute top-2 right-2 z-10"
                                        items={makeCopyItems(refinedCaption ?? '')}
                                    >
                                        <button
                                            type="button"
                                            class="cursor-pointer rounded p-1 text-gray-400 transition-colors hover:bg-gray-700 hover:text-gray-200"
                                            title="Copy"
                                            aria-label="Copy refined result"
                                            onclick={() => handleCopy(refinedCaption ?? '', false)}
                                        >
                                            {#if copiedKey !== null}
                                                <SvgCheck class="h-4 w-4 text-success" />
                                            {:else}
                                                <SvgCopy class="h-4 w-4" />
                                            {/if}
                                        </button>
                                    </ContextMenu>
                                {/if}
                                <div class="max-h-60 overflow-y-auto">
                                    {#if isEditingRefined}
                                        <!-- svelte-ignore a11y_autofocus -->
                                        <!-- Edit mode is user-initiated; focusing the input avoids an extra click. -->
                                        <textarea
                                            id="refine-result"
                                            bind:value={refinedCaption}
                                            class="w-full resize-none bg-transparent p-3 font-mono text-sm whitespace-pre-wrap text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
                                            rows="6"
                                            disabled={isAccepting}
                                            autofocus
                                        ></textarea>
                                    {:else}
                                        <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
                                        <!-- The refined caption can exceed the visible area; tabindex lets keyboard users scroll it. -->
                                        <pre
                                            class="p-3 pr-10 font-mono text-sm whitespace-pre-wrap text-gray-200"
                                            role="region"
                                            aria-label="Refined caption"
                                            tabindex="0">{refinedCaption}</pre>
                                    {/if}
                                </div>
                            </div>
                        </div>
                    {/if}

                    {#if isRefining}
                        <div
                            class="absolute inset-0 z-10 flex items-center justify-center gap-2 bg-black/60 text-sm text-gray-300"
                        >
                            <SvgSpinner class="h-4 w-4 animate-spin" />
                            <span>Refining…</span>
                        </div>
                    {/if}
                </div>

                <ActionBar>
                    {#if isRefining}
                        <!-- In-flight cancel: aborts the request and stops the
                             server-side job. Distinct from closing the dialog,
                             which would also tear down the dialog state. -->
                        <ActionBarItem
                            onclick={handleCancelRefine}
                            icon={SvgClose}
                            variant="secondary"
                        >
                            Cancel
                        </ActionBarItem>
                    {:else if isEditingRefined}
                        <ActionBarItem
                            onclick={handleUpdateRefined}
                            disabled={isAccepting}
                            icon={SvgCheck}
                            variant="primary"
                        >
                            Update
                        </ActionBarItem>
                    {:else}
                        <!-- Hidden when there's no result to act on. -->
                        {#if refinedCaption !== null}
                            <ActionBarItem
                                onclick={handleEditRefined}
                                disabled={isAccepting}
                                icon={SvgEdit}
                            >
                                Edit
                            </ActionBarItem>
                        {/if}
                        <!-- Refine — plain button (not ActionBarItem) so we can show
                             a spinner while submitting. Styled to match the surrounding
                             primary ActionBarItem. Takes the remaining width so it
                             reads as the dominant action. -->
                        <div class="flex min-w-0 flex-1 items-center justify-center">
                            <button
                                type="submit"
                                class="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-accent transition-colors hover:bg-gray-700 hover:text-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                                disabled={isRefining || !feedback.trim() || !caption.trim()}
                            >
                                <SvgRefresh class="h-4 w-4 shrink-0" />
                                Refine
                            </button>
                        </div>
                        {#if refinedCaption !== null}
                            <ActionBarItem
                                onclick={handleAccept}
                                disabled={isAccepting}
                                icon={SvgCheck}
                                variant="primary"
                            >
                                {isAccepting ? 'Applying…' : 'Accept'}
                            </ActionBarItem>
                        {/if}
                    {/if}
                </ActionBar>
            </Card>
        </div>

        <!-- Error -->
        {#if error}
            <div class="alert-error mb-4">{error}</div>
        {/if}
    </form>
</Dialog>
