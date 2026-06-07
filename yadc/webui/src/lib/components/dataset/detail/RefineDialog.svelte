<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { currentlyCaptioning, imageRefined, consumeImageRefined } from '$lib/stores/events';
    import { refineCaption as refineCaptionAction } from '$lib/stores/caption';
    import { captionOptions } from '$lib/stores/caption';
    import { friendlyErrorMessage } from '$lib/api';
    import { PasswordPromptCancelled } from '$lib/stores/passwordPrompt';
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

    // --- Local state ---
    let caption: string = $state('');
    let feedback: string = $state('');
    let isRefining: boolean = $state(false);
    let isAccepting: boolean = $state(false);
    let refinedCaption: string | null = $state(null);
    let error: string | null = $state(null);

    let activeDraftName = $derived($captionOptions?.draft?.trim() || '');

    let sourceLabel = $derived(source === 'draft' ? `draft "${draftName}"` : 'caption');

    let wasCaptioning = $state(false);

    // Reset state when dialog opens; try to restore a server-side result
    $effect(() => {
        if (open) {
            caption = currentCaption;
            feedback = '';
            refinedCaption = null;
            error = null;
            isRefining = false;
            isAccepting = false;
            wasCaptioning = false;

            // Check if the server already has a refine result for this image
            fetchRefineResult(datasetName, item.id, source, draftName)
                .then((result) => {
                    if (result !== null) {
                        refinedCaption = result;
                    }
                })
                .catch(() => {
                    // Ignore 404s and other errors silently
                });
        }
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
    //
    // FIXME: Retry after a successful refine doesn't clear the server-side
    // result, so fetchRefineResult (called on dialog open) would return the
    // stale result. This is mitigated because the dialog stays open across
    // retries, but re-opening the dialog for the same image would show the
    // old result until the new refine completes.
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

    async function handleRefine() {
        if (!feedback.trim()) {
            return;
        }
        error = null;
        refinedCaption = null;
        isRefining = true;
        try {
            await refineCaptionAction(datasetName, item.id, feedback.trim(), caption.trim());
        } catch (e) {
            if (e instanceof PasswordPromptCancelled) {
                isRefining = false;
                return;
            }
            error = friendlyErrorMessage(e, 'Failed to refine caption');
            isRefining = false;
        }
    }

    async function handleAccept() {
        if (!refinedCaption) {
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

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !isRefining && feedback.trim()) {
            e.preventDefault();
            handleRefine();
        }
    }
</script>

<Dialog class="dialog-panel max-h-[85vh] max-w-lg overflow-y-auto" {open} {onclose}>
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="p-5" onkeydown={handleKeydown}>
        <!-- Header -->
        <div class="dialog-header">
            <h2 class="dialog-title">Refine {source === 'draft' ? 'Draft' : 'Caption'}</h2>
            <button class="btn-close" onclick={onclose} aria-label="Close">
                <SvgClose class="h-5 w-5" />
            </button>
        </div>

        <!-- Current caption (editable) -->
        <div class="mb-4">
            <label class="label" for="refine-caption"
                >{source === 'draft' ? 'Draft to refine' : 'Caption to refine'}</label
            >
            <textarea
                id="refine-caption"
                bind:value={caption}
                class="input max-h-40 resize-none font-mono text-sm"
                rows="4"
                placeholder={source === 'draft' ? 'Draft text...' : 'Current caption...'}
                disabled={isRefining}
            ></textarea>
            <p class="help-text">Edit to tweak the context sent to the model.</p>
        </div>

        <!-- Feedback -->
        <div class="mb-4">
            <label class="label" for="refine-feedback">Feedback</label>
            <textarea
                id="refine-feedback"
                bind:value={feedback}
                class="input resize-none text-sm"
                rows="3"
                placeholder="e.g. Make it shorter, add detail about the background..."
                disabled={isRefining}
            ></textarea>
            <p class="help-text">Ctrl+Enter to send</p>
        </div>

        <!-- Error -->
        {#if error}
            <div class="alert-error mb-4">{error}</div>
        {/if}

        <!-- Refined result -->
        {#if refinedCaption !== null}
            <div class="mb-4">
                <p class="label">Refined result</p>
                <div class="card">
                    <pre
                        class="max-h-48 overflow-y-auto p-3 font-mono text-sm whitespace-pre-wrap text-gray-200">{refinedCaption}</pre>
                </div>
            </div>
        {/if}

        <!-- Actions -->
        <div class="btn-bar border-t border-border pt-3">
            {#if refinedCaption !== null}
                <button
                    class="btn-secondary"
                    onclick={() => {
                        refinedCaption = null;
                    }}
                    disabled={isAccepting}
                >
                    Retry
                </button>
                <button class="btn-primary" onclick={handleAccept} disabled={isAccepting}>
                    {isAccepting ? 'Applying…' : `Accept as ${sourceLabel}`}
                </button>
            {:else}
                <button class="btn-secondary" onclick={onclose} disabled={isRefining}>
                    Cancel
                </button>
                <button
                    class="btn-primary"
                    onclick={handleRefine}
                    disabled={isRefining || !feedback.trim() || !caption.trim()}
                >
                    {#if isRefining}
                        <SvgSpinner class="mr-2 h-4 w-4 animate-spin" />
                        {activeDraftName ? 'Generating draft…' : 'Refining…'}
                    {:else}
                        Refine
                    {/if}
                </button>
            {/if}
        </div>
    </div>
</Dialog>
