<script lang="ts">
    import { SvelteMap, SvelteSet } from 'svelte/reactivity';
    import {
        cancelTaggingAction,
        currentlyTagging,
        displayTag,
        fetchCachedTagResult,
        onImageTagged,
        previewImageTags,
        saveImageTagsAction,
        tagPolicy,
        tagSettings,
        tagSingleImage,
        tagTierMap,
        tagCategoryOverrideMap,
        computeOrderedCategories,
        computeSelection,
        loadTagPolicy,
        type TagChipView,
        type TaggedEntry
    } from '$lib/stores/tagging';
    import type { ImageInfo } from '$lib/stores/dataset';
    import type { TagCustomizations, TagSaveOptions, TaggerResult } from '$lib/stores/tagging';
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
    import TagCategoryChips from '$lib/components/tagging/TagCategoryChips.svelte';

    interface Props {
        datasetName: string;
        item: ImageInfo;
        /** Called after a successful save so the host can refresh caption
         *  data (drafts / extras) and other tabs reflect the write. */
        onTagsSaved?: () => void;
    }

    let { datasetName, item, onTagsSaved }: Props = $props();

    // The result for the focused image. Backend is the source of truth:
    //   - Fetched from the LRU on mount + every image switch + every
    //     policy/threshold change + every matching SSE ``image_tagged``
    //     event (thin-event style: the SSE payload is just a "this
    //     image changed" signal; we refetch so the viewer's current
    //     policy/thresholds are applied rather than the tagging
    //     request's possibly-stale ones).
    let result = $state<TaggerResult | undefined>(undefined);
    let isLoadingResult = $state(false);

    // Parent abort context — read at init time, used in effects so
    // image-switch requests are aborted when the focused image changes
    // (no race where the older response overwrites the newer one).
    const parentSignal = getAbortContext();

    // Threshold / transform settings as reactive primitives. Reading
    // ``$tagSettings.X`` directly in the fetch effect below would subscribe
    // to the whole ``tagSettings`` store, so any unrelated update (e.g. the
    // user flipping the save mode / draft format in the save bar) would
    // re-trigger a full result reload — wiping manual toggles and
    // force-enabled absent starred tags. Routing through ``$derived`` lets
    // Svelte memoize on value equality: the effect only re-runs when a
    // threshold value actually changes.
    let ratingThreshold = $derived($tagSettings.ratingThreshold);
    let generalThreshold = $derived($tagSettings.generalThreshold);
    let characterThreshold = $derived($tagSettings.characterThreshold);
    let replaceUnderscores = $derived($tagSettings.replaceUnderscores);
    let perTagThresholds = $derived($tagSettings.perTagThresholds);
    let perTagColumn = $derived($tagSettings.perTagColumn);
    // Policy lists are read reactively (not via ``get``) so toggling a
    // tag in the always-add / banned Settings re-fetches the cached
    // result — the policy is a backend read-time transform, so the
    // new view shows up without re-tagging. Frozen-snapshot copies so
    // the effect's dependency is the list identity, not deep contents.
    // Policy is now per-dataset; the mirror store tracks which dataset
    // the current value belongs to via ``$tagPolicy.datasetName`` —
    // a non-null / matching value confirms the policy is loaded for
    // this image's dataset.
    let alwaysAdd = $derived(Object.freeze([...$tagPolicy.alwaysAdd]));
    let banned = $derived(Object.freeze([...$tagPolicy.banned]));
    let policyLoadedFor = $derived($tagPolicy.datasetName);

    // Load the per-dataset policy on mount and on dataset switch.
    // Idempotent for concurrent callers — :func:`loadTagPolicy` is
    // wrapped in ``debounce({ delay: 0 })`` so the underlying fetch is
    // single-flight, and the entry is finalized on settle so a later
    // call (e.g. re-navigating back to this dataset) starts a fresh
    // fetch + re-runs the ``tagPolicy.set`` side effect.
    $effect(() => {
        void datasetName;
        loadTagPolicy(datasetName).catch((e) => {
            if ((e as Error).name !== 'AbortError') {
                toast.error('Failed to load tag policy for this dataset', {
                    details: [friendlyErrorMessage(e, 'request failed')]
                });
            }
        });
    });

    // Bumped by the SSE handler to trigger a refetch of the cached
    // result without touching the focused image. Acts as a reactive
    // dependency the fetch effect reads — Svelte re-runs the effect on
    // each bump. See the thin-events todo: ``image_tagged`` carries a
    // fat payload, but we deliberately ignore it here and refetch so a
    // policy change since the tag isn't undone by a stale payload.
    let resultNonce = $state(0);
    // Plain (non-reactive) mirror of the focused image id so the fetch
    // effect can tell a real focus switch (clear + show loading) apart
    // from a same-image refresh (SSE / policy edit: keep the stale
    // result visible until the fresh one lands, no flash). Initial -1
    // is a sentinel — image ids are positive auto-increment.
    let focusedImageId: number = -1;

    $effect(() => {
        // Re-fetch when the focused image, thresholds, or policy change,
        // or when an SSE ``image_tagged`` event bumps the nonce. Aborts
        // the previous request so an in-flight fetch for the old image
        // can't land after we've already started showing the new image.
        // Passes the current thresholds + replace_underscores so the
        // read lands in the same cache bucket as the original POST /tag
        // write. Policy is resolved server-side per dataset — a refetch
        // is needed to surface a policy change, since the cache slot
        // itself is unchanged. Only clears + shows the loading state on
        // a real focus switch — a same-image refresh (policy edit / SSE)
        // keeps the stale result visible until the fresh one lands, so
        // there's no flash.
        void resultNonce;
        void alwaysAdd;
        void banned;
        void policyLoadedFor;
        void ratingThreshold;
        void generalThreshold;
        void characterThreshold;
        void replaceUnderscores;
        void perTagThresholds;
        void perTagColumn;

        const imageId = item.id;
        const controller = linkedController(parentSignal);
        if (imageId !== focusedImageId) {
            focusedImageId = imageId;
            result = undefined;
            isLoadingResult = true;
        }

        (async () => {
            try {
                result =
                    (await fetchCachedTagResult(
                        datasetName,
                        imageId,
                        {
                            rating_threshold: ratingThreshold,
                            general_threshold: generalThreshold,
                            character_threshold: characterThreshold,
                            replace_underscores: replaceUnderscores,
                            per_tag_thresholds: perTagThresholds,
                            per_tag_column: perTagThresholds ? perTagColumn : null
                        },
                        controller.signal
                    )) ?? undefined;

                isLoadingResult = false;
            } catch (e) {
                if ((e as Error).name !== 'AbortError') {
                    result = undefined;
                    isLoadingResult = false;
                }
            }
        })();

        return () => controller.abort();
    });

    $effect(() => {
        // Subscribe to per-image tagged events (SSE + future paths).
        // Thin-event handling: ignore the fat payload and bump the
        // nonce so the fetch effect refetches with the viewer's current
        // policy / thresholds applied. Filters to the focused image —
        // other tabs/images are unaffected. Unsubscribes on destroy.
        return onImageTagged((ds, id) => {
            if (ds === datasetName && id === item.id) {
                resultNonce++;
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

    // User-added tags that aren't in the model output. ``name → category``.
    // Cleared on every fresh result (re-tagging the image discards the
    // user's additions — the new model output is the new source of truth).
    // Toggling a custom chip off leaves the entry here so it can be
    // re-enabled; the ``Clear`` button or wiping the result removes it.
    const customTags = new SvelteMap<string, string>();

    function repopulateFromResult(r: TaggerResult) {
        enabled.clear();
        customTags.clear();
        for (const tags of Object.values(r.categories)) {
            for (const t of tags) {
                enabled.add(t);
            }
        }
    }

    /** Restore the user's saved selection from a cached result's
     *  ``customizations``. Translates the persisted ``disabled`` /
     *  ``customTags`` (category → names) shape back into the internal
     *  ``enabled`` set + ``customTags`` (name → category) map. Falls back
     *  to :func:`repopulateFromResult` when none were stored. */
    function restoreFromCustomizations(r: TaggerResult, c: TagCustomizations) {
        enabled.clear();
        customTags.clear();
        const disabledSet = new Set(c.disabled);
        // Model tags: all on except those explicitly disabled.
        for (const tags of Object.values(r.categories)) {
            for (const t of tags) {
                if (!disabledSet.has(t)) {
                    enabled.add(t);
                }
            }
        }
        // Custom additions: invert category → names into name → category.
        for (const [category, names] of Object.entries(c.custom_tags)) {
            for (const name of names) {
                customTags.set(name, category);
                enabled.add(name);
            }
        }
    }

    /** Register a tag added through the per-category ``TagInput``. The chip
     *  is auto-enabled (the user just decided it should exist), and a
     *  synthetic 100% score is layered in :func:`computeSelection` so the
     *  saved output reflects the user's confidence.
     *
     *  The prune grid works in display-form identities (model tags,
     *  ``customTags``, ``enabled`` all keyed on display), so the incoming
     *  canonical entry is projected to display here via :func:`displayTag`.
     *  Free-text entries (``canonical_form: false``) and kaomojis pin
     *  their literal identity; catalog picks follow the user's
     *  ``replaceUnderscores`` preference. */
    function addCustomTag(category: string, entry: TaggedEntry) {
        const display = displayTag(entry, $tagSettings.replaceUnderscores);
        if (result && display in result.tags) {
            // Re-enable a model tag that had been pruned — don't double-
            // register it as a custom tag (it's not custom, just toggled).
            enabled.add(display);
            return;
        }
        customTags.set(display, category);
        enabled.add(display);
    }

    /** Drop a user-added custom tag from both the map and the enabled set.
     *  Triggered by the inline × button on a custom chip — the chip body
     *  click still toggles on/off like a model tag, so the user has both
     *  "hide temporarily" and "remove entirely" affordances. */
    function removeCustomTag(tag: string) {
        customTags.delete(tag);
        enabled.delete(tag);
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
        if (r.customizations) {
            restoreFromCustomizations(r, r.customizations);
        } else {
            repopulateFromResult(r);
        }
    });

    // Display layout for the prune grid — starred tags pulled to their
    // section's lead, absent starred force-shown, non-starred kept in their
    // natural category. See :func:`computeOrderedCategories` for the rules;
    // it's extracted so the routing logic is unit-testable.
    let orderedCategories = $derived(
        result
            ? computeOrderedCategories(result, customTags, $tagTierMap, $tagCategoryOverrideMap)
            : []
    );

    function toggleTag(tag: string) {
        if (enabled.has(tag)) {
            enabled.delete(tag);
        } else {
            enabled.add(tag);
        }
    }

    function selectAll(chips: TagChipView[]) {
        for (const c of chips) {
            enabled.add(c.tag);
        }
    }

    function clearAll(name: string, chips: TagChipView[]) {
        for (const c of chips) {
            enabled.delete(c.tag);
        }
        // Drop custom tags assigned to this category too — the user is
        // saying "remove everything in this category". Absent starred tags
        // (not custom) just get disabled above and stay force-shown; only
        // real custom registrations are dropped.
        for (const [t, cat] of [...customTags]) {
            if (cat === name) {
                customTags.delete(t);
                enabled.delete(t);
            }
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

    /** Compute the saved pruned result and the persisted customization
     *  snapshot in one pass. The routing / absent-starred / disabled rules
     *  live in :func:`computeSelection` (extracted so they're testable and
     *  can't drift between display, save, and persist). */
    function buildSelection() {
        if (!result) {
            return undefined;
        }
        return computeSelection(result, enabled, customTags, $tagTierMap, $tagCategoryOverrideMap);
    }

    async function handleSave() {
        const sel = buildSelection();
        if (!sel) {
            return;
        }
        isSaving = true;
        try {
            await saveImageTagsAction(datasetName, item.id, sel.pruned);
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

    const debouncedFetch = deferred(
        async (args: {
            pruned: TaggerResult;
            opts: TagSaveOptions;
            thresholds: {
                rating_threshold: number | null;
                general_threshold: number | null;
                character_threshold: number | null;
                per_tag_thresholds: boolean | null;
                per_tag_column: string | null;
            };
            customizations: TagCustomizations;
        }) => {
            inflightController?.abort();
            const controller = linkedController(parentSignal);
            inflightController = controller;
            previewLoading = true;
            try {
                previewText = await previewImageTags(
                    datasetName,
                    item.id,
                    args.pruned,
                    args.opts,
                    {
                        ...args.thresholds,
                        customizations: args.customizations
                    },
                    controller.signal
                );
            } catch (e) {
                if ((e as Error).name !== 'AbortError') {
                    previewText = '';
                }
            } finally {
                previewLoading = false;
            }
        },
        250
    );

    $effect(() => {
        // Dependencies: result, saveMode, draftFormat, and the enabled /
        // custom sets (read via buildSelection so toggles retrigger). The
        // snapshot also doubles as the customization payload persisted to
        // the backend cache, so the selection survives navigating away and
        // back.
        const sel = buildSelection();
        if (!sel) {
            previewText = '';
            return;
        }
        const opts = buildSaveOptions();
        debouncedFetch({
            pruned: sel.pruned,
            opts,
            thresholds: {
                rating_threshold: ratingThreshold,
                general_threshold: generalThreshold,
                character_threshold: characterThreshold,
                per_tag_thresholds: perTagThresholds,
                per_tag_column: perTagThresholds ? perTagColumn : null
            },
            customizations: sel.customizations
        });
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
            {#each orderedCategories as cat (cat.name)}
                {@const categoryName = cat.name}
                <TagCategoryChips
                    category={categoryName}
                    chips={cat.chips}
                    tags={result.categories[categoryName] ?? []}
                    {customTags}
                    {enabled}
                    tiers={$tagTierMap}
                    ontoggle={toggleTag}
                    onremovecustom={removeCustomTag}
                    onaddcustom={(entry) => addCustomTag(categoryName, entry)}
                    onselectall={() => selectAll(cat.chips)}
                    onclear={() => clearAll(categoryName, cat.chips)}
                />
            {/each}
        </div>
    {/if}
</div>
