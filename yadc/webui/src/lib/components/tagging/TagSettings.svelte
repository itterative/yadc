<script lang="ts">
    import {
        tagOptions as tagOptionsStore,
        tagSettings,
        TAG_TIERS,
        CANONICAL_THRESHOLDS,
        previewTagFormats,
        alwaysAddList,
        bannedList,
        addToAlwaysAdd,
        removeFromAlwaysAdd,
        addToBanned,
        removeFromBanned,
        TAG_CATEGORIES
    } from '$lib/stores/tagging';
    import type { TagSaveOptions, TaggerResult } from '$lib/stores/tagging';
    import { activeTagger, ensureActiveTaggerLoaded } from '$lib/stores/tagging';
    import Checkbox from '$lib/components/ui/Checkbox.svelte';
    import Slider from '$lib/components/ui/Slider.svelte';
    import PolicyList from './PolicyList.svelte';
    import { deferred } from '$lib/async';
    import { get } from 'svelte/store';
    import { getAbortContext, linkedController } from '$lib/abort';

    // --- Threshold overrides (null = server default) ---

    let ratingThreshold: number | null = $state(null);
    let generalThreshold: number | null = $state(null);
    let characterThreshold: number | null = $state(null);
    let replaceUnderscores = $state(false);
    let perTagThresholds = $state(false);
    let perTagColumn = $state('best_threshold');
    let supportsPerTag = $state(false);
    let perTagColumns: string[] = $state([]);

    const PER_TAG_COLUMN_LABELS: Record<string, string> = {
        best_threshold: 'Best F1 threshold',
        best_recall: 'Best recall (higher)'
    };

    // --- Per-tag support (from the active-tagger store) ---
    // Capability comes from ``GET /api/tagger/active`` via the shared
    // store, which re-fetches whenever the ``tagger_status`` SSE stream
    // settles — so a swap (including minutes-long first-run HF
    // downloads) surfaces here in every tab without polling.

    $effect(() => {
        ensureActiveTaggerLoaded().catch(() => {});
    });

    $effect(() => {
        const active = $activeTagger?.active;
        supportsPerTag = active?.has_per_tag_thresholds ?? false;
        perTagColumns = active?.per_tag_columns ?? [];
        if (!supportsPerTag) {
            perTagThresholds = false;
            perTagColumn = 'best_threshold';
        } else if (perTagColumns.length > 0 && !perTagColumns.includes(perTagColumn)) {
            perTagColumn = perTagColumns[0];
        }
    });

    // --- Save options ---

    let saveMode: TagSaveOptions['mode'] = $state('none');
    let draftName = $state('tags');
    let draftFormat = $state('comma');
    let overwrite = $state(false);

    // --- Diff dots (vs canonical wd-tagger defaults) ---

    function diffDot(current: number | null, canonical: number): boolean {
        return current !== null && Math.abs(current - canonical) > 1e-6;
    }

    let ratingOverridden = $derived(diffDot(ratingThreshold, CANONICAL_THRESHOLDS.rating));
    let generalOverridden = $derived(diffDot(generalThreshold, CANONICAL_THRESHOLDS.general));
    let characterOverridden = $derived(diffDot(characterThreshold, CANONICAL_THRESHOLDS.character));

    // --- Load persisted settings on mount ---

    let dataLoaded = $state(false);

    $effect(() => {
        if (!dataLoaded) {
            dataLoaded = true;
            const saved = get(tagSettings);
            ratingThreshold = saved.ratingThreshold;
            generalThreshold = saved.generalThreshold;
            characterThreshold = saved.characterThreshold;
            replaceUnderscores = saved.replaceUnderscores;
            saveMode = saved.saveMode;
            draftName = saved.draftName || 'tags';
            draftFormat = saved.draftFormat || 'comma';
            overwrite = saved.overwrite;
            perTagThresholds = saved.perTagThresholds;
            perTagColumn = saved.perTagColumn || 'best_threshold';
        }
    });

    // --- Persist settings (debounced) ---

    function _buildSettings() {
        return {
            $version: 4,
            ratingThreshold,
            generalThreshold,
            characterThreshold,
            replaceUnderscores,
            saveMode,
            draftName,
            draftFormat,
            overwrite,
            perTagThresholds,
            perTagColumn
        };
    }

    const persistSettings = deferred(() => {
        tagSettings.set(_buildSettings());
    }, 300);

    $effect(() => {
        if (!dataLoaded) {
            return;
        }
        void ratingThreshold;
        void generalThreshold;
        void characterThreshold;
        void replaceUnderscores;
        void saveMode;
        void draftName;
        void draftFormat;
        void overwrite;
        void perTagThresholds;
        void perTagColumn;
        persistSettings();
    });

    // --- Assemble options for the action layer ---

    let assembledSave: TagSaveOptions = $derived({
        mode: saveMode,
        draft_name: draftName || 'tags',
        draft_format: draftFormat || 'comma',
        overwrite
    });

    let _assembledOptions = $derived({
        rating_threshold: ratingThreshold ?? undefined,
        general_threshold: generalThreshold ?? undefined,
        character_threshold: characterThreshold ?? undefined,
        replace_underscores: replaceUnderscores,
        per_tag_thresholds: perTagThresholds,
        per_tag_column: perTagThresholds ? perTagColumn : undefined,
        save: assembledSave
    });

    // Keep the shared store in sync so single-image tagging from ImageDetail
    // uses the current thresholds, and so the panel's Start button picks
    // them up. Stays mounted even when the Customize tab is active (inactive
    // ``Tab``s are ``hidden`` but not destroyed).
    $effect(() => {
        tagOptionsStore.set(_assembledOptions);
    });

    // --- Format preview (draft / extras) ---
    // Shows what the selected save would produce from fixed mock data, via
    // the context-free backend preview (formatters are server-side — not
    // duplicated client-side). Refetches (debounced) on save-option change;
    // stale requests are aborted. Hidden for ``none`` (nothing to show).
    const MOCK_RESULT: TaggerResult = {
        tags: {
            general: 0.95,
            '1girl': 0.99,
            solo: 0.9,
            'simple background': 0.85,
            'white background': 0.8
        },
        categories: {
            rating: ['general'],
            general: ['1girl', 'solo', 'simple background', 'white background']
        }
    };

    let previewText = $state('');
    let previewLoading = $state(false);

    // Parent abort context — read at init time, used in effects.
    const parentSignal = getAbortContext();

    $effect(() => {
        // Dependencies: saveMode, draftName, draftFormat.
        const opts = assembledSave;
        if (opts.mode === 'none') {
            previewText = '';
            previewLoading = false;
            return;
        }
        // Inputs are dropdowns, not a stream of edits — no debounce needed.
        // Each effect run links a fresh controller to the parent context and
        // the cleanup aborts it before the next run, so a fast flip doesn't
        // paint stale output from the older response.
        const controller = linkedController(parentSignal);
        previewLoading = true;
        void (async () => {
            try {
                previewText = await previewTagFormats(MOCK_RESULT, opts, controller.signal);
            } catch (e) {
                if ((e as Error).name !== 'AbortError') {
                    previewText = '';
                }
            } finally {
                previewLoading = false;
            }
        })();
        return () => controller.abort();
    });

    function resetThreshold(field: keyof typeof TAG_CATEGORIES) {
        if (field === TAG_CATEGORIES.rating) {
            ratingThreshold = null;
        } else if (field === TAG_CATEGORIES.general) {
            generalThreshold = null;
        } else {
            characterThreshold = null;
        }
    }
</script>

{#snippet thresholdRow(
    id: string,
    label: string,
    threshold: number | null,
    canonical: number,
    overridden: boolean,
    set: (v: number) => void,
    reset: () => void
)}
    <div>
        <div class="mb-1 flex items-center justify-between">
            <label class="label" for={id}>{label}</label>
            {#if overridden}
                <button class="mr-2 text-xs text-accent hover:text-accent-hover" onclick={reset}>
                    reset
                </button>
            {/if}
        </div>
        <div class="flex min-w-0 items-center gap-3">
            <div class="min-w-0 flex-1">
                <Slider
                    {id}
                    value={threshold ?? canonical}
                    min={0}
                    max={1}
                    step={0.01}
                    ariaLabel={`${label} threshold`}
                    oninput={set}
                />
            </div>
            <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                placeholder={String(canonical)}
                value={threshold ?? ''}
                class="input-sm w-20 shrink-0"
                aria-label={`${label} threshold value`}
                oninput={(e) => {
                    const raw = (e.target as HTMLInputElement).value;
                    if (raw === '') {
                        reset();
                    } else {
                        set(Number(raw));
                    }
                }}
            />
        </div>
    </div>
{/snippet}

<div class="space-y-5">
    <!-- ═══ Section: Thresholds ═══ -->
    <section class="space-y-2">
        <h3 class="section-heading">Thresholds</h3>
        <p class="text-xs text-gray-500">
            Minimum confidence for a tag to be kept. Blank number field (or the slider's rest
            position) = server default, shown as the placeholder. Rating is categorical — keep it
            low.
        </p>

        <div class="space-y-3">
            {#if supportsPerTag}
                <div class="flex items-start gap-2">
                    <Checkbox id="tag-per-tag-thresholds" bind:checked={perTagThresholds} />
                    <div>
                        <label class="cursor-pointer text-sm" for="tag-per-tag-thresholds">
                            Use per-tag thresholds (from CSV)
                        </label>
                        <p class="text-xs text-gray-500">
                            When the model's CSV contains per-tag optimal thresholds (e.g.
                            animetimm), use those instead of the three global category thresholds
                            below.
                        </p>
                    </div>
                </div>
                {#if perTagThresholds}
                    <div>
                        <label class="label mb-1 block" for="tag-per-tag-column"
                            >Threshold column</label
                        >
                        <select
                            id="tag-per-tag-column"
                            class="input"
                            value={perTagColumn}
                            onchange={(e) =>
                                (perTagColumn = (e.currentTarget as HTMLSelectElement).value)}
                        >
                            {#each perTagColumns.length > 0 ? perTagColumns : ['best_threshold', 'best_recall'] as col (col)}
                                <option value={col}>{PER_TAG_COLUMN_LABELS[col] ?? col}</option>
                            {/each}
                        </select>
                    </div>
                {/if}
            {/if}
        </div>

        {#if !perTagThresholds}
            <div class="space-y-3">
                {@render thresholdRow(
                    'tag-rating-threshold',
                    'Rating',
                    ratingThreshold,
                    CANONICAL_THRESHOLDS.rating,
                    ratingOverridden,
                    (v) => (ratingThreshold = v),
                    () => resetThreshold(TAG_CATEGORIES.rating)
                )}
                {@render thresholdRow(
                    'tag-general-threshold',
                    'General',
                    generalThreshold,
                    CANONICAL_THRESHOLDS.general,
                    generalOverridden,
                    (v) => (generalThreshold = v),
                    () => resetThreshold(TAG_CATEGORIES.general)
                )}
                {@render thresholdRow(
                    'tag-character-threshold',
                    'Character',
                    characterThreshold,
                    CANONICAL_THRESHOLDS.character,
                    characterOverridden,
                    (v) => (characterThreshold = v),
                    () => resetThreshold(TAG_CATEGORIES.character)
                )}
            </div>
        {/if}

        <!-- Underscore replacement -->
        <div class="mt-3 flex items-start gap-2">
            <Checkbox id="tag-replace-underscores" bind:checked={replaceUnderscores} />
            <div>
                <label class="cursor-pointer text-sm" for="tag-replace-underscores">
                    Replace underscores with spaces
                </label>

                <p class="text-xs text-gray-500">
                    WD-tagger tags ship underscored (e.g. <code>long_hair</code>). On, they read as
                    <code>long hair</code>
                    in results and saves. Kaomojis (e.g. <code>^_^</code>) are always preserved.
                </p>
            </div>
        </div>
    </section>

    <!-- ═══ Section: Tags ═══ -->
    <section class="space-y-4">
        <PolicyList
            title="Always add"
            description="Forced into every tagged image's result at score 1.0. Use this for tags you want on every image in the dataset (e.g. quality markers)."
            list={alwaysAddList}
            tier={TAG_TIERS.starred}
            onadd={addToAlwaysAdd}
            onremove={removeFromAlwaysAdd}
            activeFillClass="border-amber-400/60 bg-amber-400/20 text-amber-300"
        />

        <PolicyList
            title="Banned"
            description="Removed from every tagged image's result entirely. Use this for tags that should never appear regardless of the model's confidence (e.g. content you don't want in the dataset)."
            list={bannedList}
            tier={TAG_TIERS.undesired}
            onadd={addToBanned}
            onremove={removeFromBanned}
            activeFillClass="border-rose-400/60 bg-rose-400/20 text-rose-300"
        />
    </section>

    <!-- ═══ Section: Save ═══ -->
    <section class="space-y-2">
        <h3 class="section-heading">Save</h3>
        <p class="text-xs text-gray-500">Where to write each image's tags during the batch run.</p>
        <div class="space-y-3">
            <div>
                <label class="label mb-1 block" for="tag-save-mode-batch">Destination</label>
                <select
                    id="tag-save-mode-batch"
                    class="input"
                    value={saveMode}
                    onchange={(e) =>
                        (saveMode = (e.currentTarget as HTMLSelectElement)
                            .value as TagSaveOptions['mode'])}
                >
                    <option value="none">Preview (tag only)</option>
                    <option value="draft">Draft</option>
                    <option value="extras">Extras (TOML)</option>
                </select>
                {#if saveMode === 'none'}
                    <p class="mt-2 text-xs text-gray-500">
                        Writes nothing. Results live only in each image's <strong>Tags</strong> tab —
                        held in memory, so they're lost on reload. Useful for auditing confidence before
                        committing to a save mode.
                    </p>
                {:else if saveMode === 'draft'}
                    <p class="mt-2 text-xs text-gray-500">
                        Writes formatted tag text to a draft sidecar file per image (e.g.
                        <code>{`<id>.${assembledSave.draft_name}.draft~`}</code>). Used as the
                        training caption for image-generation fine-tuning.
                    </p>
                {:else}
                    <p class="mt-2 text-xs text-gray-500">
                        Merges a <code>[tags]</code> section into each image's TOML, preserving other
                        keys.
                    </p>
                {/if}
            </div>
            {#if saveMode !== 'none'}
                <div class="flex items-start gap-2">
                    <Checkbox id="tag-overwrite-batch" bind:checked={overwrite} />
                    <div>
                        <label class="cursor-pointer text-sm" for="tag-overwrite-batch">
                            Overwrite existing tags
                        </label>
                        <p class="text-xs text-gray-500">
                            When off (default), images that already have a saved {saveMode ===
                            'draft'
                                ? `"${assembledSave.draft_name}" draft`
                                : '[tags] section in TOML'} are skipped entirely — no tag, no write. Turn
                            on to re-tag and rewrite every image.
                        </p>
                    </div>
                </div>
            {/if}
            {#if saveMode === 'draft'}
                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="label mb-1 block" for="tag-draft-name-batch">Draft name</label
                        >
                        <input
                            id="tag-draft-name-batch"
                            class="input"
                            value={draftName}
                            oninput={(e) => (draftName = (e.target as HTMLInputElement).value)}
                        />
                    </div>
                    <div>
                        <label class="label mb-1 block" for="tag-draft-format-batch">Format</label>
                        <select
                            id="tag-draft-format-batch"
                            class="input"
                            value={draftFormat}
                            onchange={(e) =>
                                (draftFormat = (e.currentTarget as HTMLSelectElement).value)}
                        >
                            <option value="comma">Comma list</option>
                            <option value="structured">Structured</option>
                            <option value="scored">Scored (with confidence)</option>
                        </select>
                    </div>
                </div>
            {/if}
            {#if saveMode === 'draft' || saveMode === 'extras'}
                {@const _previewText = previewText || '(empty)'}
                <div class="rounded-md border border-border bg-gray-900/40 p-2">
                    <div class="mb-1 flex items-center gap-1.5 text-xs text-gray-500">
                        <span>Preview</span>
                        {#if previewLoading}
                            <span class="animate-pulse">…</span>
                        {/if}
                    </div>
                    <pre
                        class="overflow-x-auto font-mono text-xs wrap-break-word whitespace-pre-wrap text-gray-300">{_previewText}</pre>
                </div>
            {/if}
        </div>
    </section>
</div>

<style lang="postcss">
    @reference "tailwindcss";

    code {
        @apply text-gray-400;
    }
</style>
