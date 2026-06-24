<script lang="ts">
    import { SvelteSet } from 'svelte/reactivity';
    import {
        cancelTaggingAction,
        currentlyTagging,
        fetchCachedTagResult,
        onImageTagged,
        previewImageTags,
        saveImageTagsAction,
        tagSettings,
        tagSingleImage
    } from '$lib/stores/tagging';
    import type { ImageInfo } from '$lib/stores/dataset';
    import type { TagSaveOptions, TaggerResult } from '$lib/stores/tagging';
    import Card from '$lib/components/ui/Card.svelte';
    import ActionBar from '$lib/components/ui/ActionBar.svelte';
    import ActionBarItem from '$lib/components/ui/ActionBarItem.svelte';
    import SvgTag from '$lib/icons/SvgTag.svelte';
    import SvgCheck from '$lib/icons/SvgCheck.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import { toast } from '$lib/stores/toasts';
    import { friendlyErrorMessage } from '$lib/api';
    import { deferred } from '$lib/async';
    import { getAbortContext, linkedController } from '$lib/abort';
    import SvgSparkle from '$lib/icons/SvgSparkle.svelte';

    interface Props {
        datasetName: string;
        item: ImageInfo;
        /** Called after a successful save so the host can refresh caption
         *  data (drafts / extras) and other tabs reflect the write. */
        onTagsSaved?: () => void;
    }

    let { datasetName, item, onTagsSaved }: Props = $props();

    // The result for the focused image. Backend is the source of truth:
    //   - Fetched from the LRU on mount + every image switch
    //   - Updated in place by the Tag click response
    //   - Updated in place by SSE ``image_tagged`` events that match
    //     the focused image (registered via ``onImageTagged``)
    let result = $state<TaggerResult | undefined>(undefined);
    let isLoadingResult = $state(false);

    // Parent abort context — read at init time, used in effects so
    // image-switch requests are aborted when the focused image changes
    // (no race where the older response overwrites the newer one).
    const parentSignal = getAbortContext();

    $effect(() => {
        // Re-fetch when the focused image changes. Aborts the previous
        // request so an in-flight fetch for the old image can't land
        // after we've already started showing the new image. Pass the
        // current thresholds + replace_underscores so the read lands
        // in the same cache bucket as the original POST /tag write.
        const controller = linkedController(parentSignal);
        isLoadingResult = true;
        result = undefined;
        fetchCachedTagResult(
            datasetName,
            item.id,
            {
                rating_threshold: $tagSettings.ratingThreshold,
                general_threshold: $tagSettings.generalThreshold,
                character_threshold: $tagSettings.characterThreshold,
                replace_underscores: $tagSettings.replaceUnderscores
            },
            controller.signal
        )
            .then((r) => {
                result = r ?? undefined;
                isLoadingResult = false;
            })
            .catch((e) => {
                if ((e as Error).name !== 'AbortError') {
                    result = undefined;
                    isLoadingResult = false;
                }
            });
        return () => controller.abort();
    });

    $effect(() => {
        // Subscribe to per-image tagged events (SSE + future paths).
        // Filters to the focused image — other tabs/images are
        // unaffected. Unsubscribes on destroy.
        return onImageTagged((ds, id, r) => {
            if (ds === datasetName && id === item.id) {
                result = r;
            }
        });
    });

    let isTagging = $derived.by(() => {
        for (const entry of $currentlyTagging) {
            if (entry.dataset_name === datasetName && entry.image_id === item.id) {
                return true;
            }
        }
        return false;
    });

    // True while a cancel (kill-the-tagger) request is in-flight. Driven by
    // the synchronous cancel request itself (the endpoint holds until the
    // kill took effect), not the SSE store — so the "Cancelling…" window is
    // always exactly the request duration, never racy on event timing.
    let isCancelling = $state(false);

    async function handleCancel() {
        isCancelling = true;
        try {
            await cancelTaggingAction(datasetName, item.id);
        } finally {
            isCancelling = false;
        }
    }

    // Working set of kept tags. ``SvelteSet`` is reactive, so in-place
    // mutations (add/delete/clear) drive updates — never reassign it.
    // Reset to "all categorized tags enabled" whenever a fresh result
    // arrives for this image.
    let enabled = new SvelteSet<string>();
    let lastResultSeen: TaggerResult | undefined = undefined;

    function repopulateFromResult(r: TaggerResult) {
        enabled.clear();
        for (const tags of Object.values(r.categories)) {
            for (const t of tags) {
                enabled.add(t);
            }
        }
    }

    $effect(() => {
        const r = result;
        if (r === undefined) {
            enabled.clear();
            lastResultSeen = undefined;
            return;
        }
        if (r === lastResultSeen) {
            return;
        }
        lastResultSeen = r;
        repopulateFromResult(r);
    });

    // Stable display order: the wd-tagger convention is rating / general /
    // character; any other categories sort after, alphabetically.
    const CATEGORY_ORDER = ['rating', 'character', 'general'];
    let orderedCategories = $derived.by(() => {
        if (!result) {
            return [];
        }
        const keys = Object.keys(result.categories);
        return [...keys].sort((a, b) => {
            const ia = CATEGORY_ORDER.indexOf(a);
            const ib = CATEGORY_ORDER.indexOf(b);
            if (ia !== -1 || ib !== -1) {
                return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
            }
            return a.localeCompare(b);
        });
    });

    function toggleTag(tag: string) {
        if (enabled.has(tag)) {
            enabled.delete(tag);
        } else {
            enabled.add(tag);
        }
    }

    function selectAll(category?: string) {
        const r = result;
        if (!r) {
            return;
        }

        const categoryTags = category
            ? [r.categories[category] ?? []]
            : Array(...Object.keys(r.categories)).map((c) => r.categories[c]);

        for (const tags of categoryTags) {
            for (const t of tags) {
                enabled.add(t);
            }
        }
    }

    function clearAll(category?: string) {
        if (!category) {
            enabled.clear();
            return;
        }

        if (!result) {
            return;
        }

        const tags = result.categories[category] ?? [];

        for (const t of tags) {
            enabled.delete(t);
        }
    }

    // --- Save ---

    let isSaving = $state(false);
    let saveMode = $derived.by(() => {
        // Auto-subscribe via ``$tagSettings`` so this recomputes on change
        // (a bare ``get()`` read would be one-shot and never update).
        const m = $tagSettings.saveMode;
        // 'none' isn't offered here (no point saving nothing from the
        // interactive bar) — coerce a stale or batch-persisted 'none' to
        // 'draft' so the select always has a valid option.
        return m === 'none' ? 'draft' : m;
    });
    let draftName = $derived.by(() => $tagSettings.draftName || 'tags');
    let draftFormat = $derived.by(() => $tagSettings.draftFormat || 'comma');

    let keptCount = $derived(enabled.size);

    async function handleTag() {
        try {
            // ``tagSingleImage`` returns the freshly-computed result;
            // we set it directly so the prune grid updates without
            // waiting for a round trip back to the backend LRU.
            result = await tagSingleImage(datasetName, item.id);
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to tag image'));
        }
    }

    function buildPrunedResult(): TaggerResult {
        const r = result!;
        const prunedCategories: Record<string, string[]> = {};
        const prunedTags: Record<string, number> = {};
        for (const [cat, tags] of Object.entries(r.categories)) {
            const kept = tags.filter((t) => enabled.has(t));
            if (kept.length > 0) {
                prunedCategories[cat] = kept;
            }
        }
        for (const tag of enabled) {
            if (tag in r.tags) {
                prunedTags[tag] = r.tags[tag];
            }
        }
        return { tags: prunedTags, categories: prunedCategories };
    }

    async function handleSave() {
        if (!result) {
            return;
        }
        isSaving = true;
        try {
            await saveImageTagsAction(datasetName, item.id, buildPrunedResult());
            toast.success('Tags saved');
            onTagsSaved?.();
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to save tags'));
        } finally {
            isSaving = false;
        }
    }

    // --- Save preview ---
    // Live-formats what the save would write, via a backend endpoint (the
    // formatters are a server-side registry — not duplicated client-side).
    // Refetches (debounced) whenever the pruned result or save options
    // change; stale requests are aborted so rapid tag toggles don't race.
    // ``parentSignal`` was already captured at the top of the script for
    // the result-fetch effect; reuse it here.

    let previewText = $state('');
    let previewLoading = $state(false);

    // Tracks the most recent in-flight request's controller so the next
    // debounce fire can abort it before starting a new one (avoids a race
    // where the older response overwrites the newer one). Plain closure
    // ref — not $state because nothing in the template depends on it.
    let inflightController: AbortController | null = null;

    function buildSaveOptions(): TagSaveOptions {
        return {
            mode: saveMode,
            draft_name: draftName,
            draft_format: draftFormat,
            overwrite: true
        };
    }

    const debouncedFetch = deferred(async (pruned: TaggerResult, opts: TagSaveOptions) => {
        inflightController?.abort();
        const controller = linkedController(parentSignal);
        inflightController = controller;
        previewLoading = true;
        try {
            previewText = await previewImageTags(
                datasetName,
                item.id,
                pruned,
                opts,
                controller.signal
            );
        } catch (e) {
            if ((e as Error).name !== 'AbortError') {
                previewText = '';
            }
        } finally {
            previewLoading = false;
        }
    }, 250);

    $effect(() => {
        // Dependencies: result, saveMode, draftFormat, and the enabled set
        // (read via buildPrunedResult so toggles retrigger).
        if (!result) {
            previewText = '';
            return;
        }
        const pruned = buildPrunedResult();
        const opts = buildSaveOptions();
        debouncedFetch(pruned, opts);
    });
</script>

<div class="flex h-full flex-col gap-4">
    <div>
        <h3 class="mb-1 text-sm font-medium text-gray-300">Tags</h3>
        <p class="text-xs text-gray-500">
            Run the tagger, prune the result, then save to a draft or the image TOML.
        </p>
    </div>

    {#if !result}
        <Card>
            <div class="p-6 text-center text-sm text-gray-500">
                {#if isTagging}
                    <SvgTag class="mx-auto mb-2 h-6 w-6 animate-pulse text-accent" />
                    Running the tagger…
                {:else if isLoadingResult}
                    <SvgTag class="mx-auto mb-2 h-6 w-6 animate-pulse text-gray-600" />
                    Checking for cached result…
                {:else}
                    <SvgTag class="mx-auto mb-2 h-6 w-6 text-gray-600" />
                    No tags yet. Click <span class="text-gray-300">Tag</span> to run the model.
                {/if}
            </div>

            <ActionBar>
                {#if isTagging || isCancelling}
                    <ActionBarItem
                        onclick={handleCancel}
                        disabled={isCancelling}
                        icon={SvgClose}
                        variant="danger"
                    >
                        {isCancelling ? 'Cancelling...' : 'Cancel'}
                    </ActionBarItem>
                {:else}
                    <ActionBarItem
                        onclick={handleTag}
                        disabled={isTagging}
                        icon={SvgSparkle}
                        variant="primary"
                    >
                        Tag
                    </ActionBarItem>
                {/if}
            </ActionBar>
        </Card>
    {:else}
        <!-- Save bar -->
        <Card class="shrink-0">
            {#if result}
                <div class="border-b border-gray-700/50 px-3 py-2">
                    <div class="mb-1 flex items-center justify-between">
                        <span class="text-xs text-gray-500">Preview</span>
                        {#if previewLoading}
                            <span class="text-xs text-gray-600">&ensp;formatting…</span>
                        {/if}
                    </div>
                    <pre class="p-2 text-sm wrap-break-word whitespace-pre-wrap text-gray-300"><code
                            >{previewText || '(empty)'}</code
                        ></pre>
                </div>
            {/if}

            <div class="flex flex-wrap items-end gap-3 p-3">
                <div class="min-w-28 flex-1">
                    <label class="label mb-1 block text-xs" for="tag-save-mode">Save to</label>
                    <select
                        id="tag-save-mode"
                        class="input"
                        value={saveMode}
                        onchange={(e) =>
                            tagSettings.update((s) => ({
                                ...s,
                                saveMode: (e.currentTarget as HTMLSelectElement).value as
                                    | 'draft'
                                    | 'extras'
                            }))}
                    >
                        <option value="draft">Draft</option>
                        <option value="extras">Extras (TOML)</option>
                    </select>
                </div>
                {#if saveMode === 'draft'}
                    <div class="min-w-24 flex-1">
                        <label class="label mb-1 block text-xs" for="tag-draft-name"
                            >Draft name</label
                        >
                        <input
                            id="tag-draft-name"
                            class="input"
                            value={draftName}
                            oninput={(e) =>
                                tagSettings.update((s) => ({
                                    ...s,
                                    draftName: (e.target as HTMLInputElement).value
                                }))}
                        />
                    </div>
                    <div class="min-w-24 flex-1">
                        <label class="label mb-1 block text-xs" for="tag-draft-format">Format</label
                        >
                        <select
                            id="tag-draft-format"
                            class="input"
                            value={draftFormat}
                            onchange={(e) =>
                                tagSettings.update((s) => ({
                                    ...s,
                                    draftFormat: (e.currentTarget as HTMLSelectElement).value
                                }))}
                        >
                            <option value="comma">Comma list</option>
                            <option value="structured">Structured</option>
                            <option value="scored">Scored (with confidence)</option>
                        </select>
                    </div>
                {/if}
            </div>
            <ActionBar>
                <ActionBarItem
                    onclick={handleSave}
                    disabled={keptCount === 0 || isSaving}
                    icon={SvgCheck}
                    variant="primary"
                >
                    {isSaving ? 'Saving…' : 'Save'}
                </ActionBarItem>
                <ActionBarItem
                    onclick={() => {
                        if (result) {
                            repopulateFromResult(result);
                        }
                    }}
                    disabled={isSaving}
                    icon={SvgClose}
                    variant="secondary"
                >
                    Reset
                </ActionBarItem>
            </ActionBar>
        </Card>

        <div>
            <h3 class="mt-4 mb-1 text-sm font-medium text-gray-300">Refine</h3>
            <p class="text-xs text-gray-500">
                You can toggle individual tags off if you want to exclude them from the final
                result.
            </p>
        </div>

        <!-- Result: grouped toggle chips -->
        <div class="flex flex-1 flex-col gap-4">
            {#each orderedCategories as cat (cat)}
                {@const categoryTags = result.categories[cat]}
                <section>
                    <div class="mb-3 flex items-center justify-end gap-3 text-xs text-gray-400">
                        <h4 class="flex-1 font-semibold text-gray-400 uppercase">
                            {cat}
                        </h4>

                        {#if categoryTags.length > 0}
                            <button
                                class="cursor-pointer hover:text-gray-200"
                                onclick={() => selectAll(cat)}>Select all</button
                            >
                            <span class="text-gray-600">·</span>
                            <button
                                class="cursor-pointer hover:text-gray-200"
                                onclick={() => clearAll(cat)}>Clear</button
                            >
                        {/if}
                    </div>

                    <div class="flex flex-wrap gap-1.5">
                        {#if categoryTags.length === 0}
                            <span class="pl-1 text-xs text-gray-400 italic">(none)</span>
                        {/if}
                        {#each categoryTags as tag (tag)}
                            {@const score = result.tags[tag] ?? 0}
                            <button
                                class="cursor-pointer rounded-md border px-2 py-1 text-xs transition-colors {enabled.has(
                                    tag
                                )
                                    ? 'border-accent/60 bg-accent/20 text-accent'
                                    : 'border-gray-700 bg-gray-800 text-gray-500 line-through opacity-60 hover:border-gray-600'}"
                                onclick={() => toggleTag(tag)}
                                title={score.toFixed(4)}
                            >
                                <span>{tag}</span>
                                <span class="ml-1.5 text-[10px] opacity-70">
                                    {Math.round(score * 100)}%
                                </span>
                            </button>
                        {/each}
                    </div>
                </section>
            {/each}
        </div>
    {/if}
</div>
