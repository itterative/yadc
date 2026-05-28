<script lang="ts">
    import EnvSelector from '$lib/components/settings/EnvSelector.svelte';
    import JinjaEditor from '$lib/components/ui/JinjaEditor.svelte';
    import Checkbox from '$lib/components/ui/Checkbox.svelte';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SvgChevronLeft from '$lib/icons/SvgChevronLeft.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import {
        templates,
        refreshTemplates,
        fetchTemplate,
        saveTemplate
    } from '$lib/stores/templates';
    import type { CaptionOptions } from '$lib/stores/captionOptions';
    import { captionSettings } from '$lib/stores/captionSettings';
    import { promptNotificationsOnce } from '$lib/notifications';
    import { fetchConfig } from '$lib/stores/configs';
    import { get } from 'svelte/store';
    import { friendlyErrorMessage } from '$lib/api';

    // --- Props ---

    interface Props {
        /** Which dataset this is for. */
        datasetName: string;
        /** The currently assembled caption options (reactive, updated as settings change). */
        currentOptions?: CaptionOptions;
        /** Whether a batch captioning job is currently running for this dataset. */
        isBatchCaptioning?: boolean;
        /** Called when the user clicks "Start Captioning". Receives the assembled options. */
        onstart?: (options: CaptionOptions) => void;
        /** Called when the user clicks "Stop Captioning". */
        onstop?: () => void;
        /** Called when the panel wants to close (e.g. after starting). */
        onclose?: () => void;
    }

    let {
        datasetName: _datasetName,
        currentOptions = $bindable(),
        isBatchCaptioning = false,
        onstart,
        onstop,
        onclose: _onclose
    }: Props = $props();

    // --- State: Environment (managed by EnvSelector via bindings) ---

    let selectedEnv = $state('default');
    let envUrl = $state('');
    let envToken = $state('');
    let envModelName = $state('');

    // --- State: Templates ---

    let selectedTemplate = $state('');
    let templateContent = $state('');
    let templateSource: 'user' | 'builtin' | '' = $state('');
    let isLoadingTemplate = $state(false);

    let isNewTemplate = $state(false);
    let newTemplateName = $state('');
    let isSavingTemplate = $state(false);
    let templateSaveError: string | null = $state(null);

    // --- State: Options ---

    let maxTokens = $state(512);
    let imageQuality: 'auto' | 'high' | 'low' = $state('auto');
    let draftName = $state('');
    let overwrite = $state(false);
    let rounds = $state(1);

    // --- State: Reasoning ---

    let reasoningEnabled = $state(false);
    let reasoningEffort: 'low' | 'medium' | 'high' = $state('low');

    // --- State: General ---

    let isLoadingTemplates = $state(false);
    let templatesError: string | null = $state(null);

    // --- Dataset config defaults ---

    interface DatasetDefaults {
        maxTokens: number;
        imageQuality: 'auto' | 'high' | 'low';
        draftName: string;
        overwrite: boolean;
        rounds: number;
        reasoningEnabled: boolean;
        reasoningEffort: 'low' | 'medium' | 'high';
        selectedTemplate: string;
    }

    const HARDCODED_DEFAULTS: DatasetDefaults = {
        maxTokens: 512,
        imageQuality: 'auto',
        draftName: '',
        overwrite: false,
        rounds: 1,
        reasoningEnabled: false,
        reasoningEffort: 'low',
        selectedTemplate: ''
    };

    let datasetDefaults: DatasetDefaults = $state({ ...HARDCODED_DEFAULTS });

    // --- Computed ---

    let effectiveModelName = $derived(envModelName.trim());
    let templateDirty = $state(false);
    let effectiveTemplateName = $derived(isNewTemplate ? newTemplateName.trim() : selectedTemplate);

    // --- Diff tracking ---

    let maxTokensOverridden = $derived(maxTokens !== datasetDefaults.maxTokens);
    let imageQualityOverridden = $derived(imageQuality !== datasetDefaults.imageQuality);
    let draftNameOverridden = $derived(draftName !== datasetDefaults.draftName);
    let overwriteOverridden = $derived(overwrite !== datasetDefaults.overwrite);
    let roundsOverridden = $derived(rounds !== datasetDefaults.rounds);
    let reasoningEnabledOverridden = $derived(
        reasoningEnabled !== datasetDefaults.reasoningEnabled
    );
    let reasoningEffortOverridden = $derived(
        reasoningEnabled && reasoningEffort !== datasetDefaults.reasoningEffort
    );
    let templateOverridden = $derived(selectedTemplate !== datasetDefaults.selectedTemplate);

    /** Human-readable labels for each overrideable field. */
    const FIELD_LABELS: Record<keyof DatasetDefaults, string> = {
        maxTokens: 'Max Tokens',
        imageQuality: 'Image Quality',
        draftName: 'Draft Name',
        overwrite: 'Overwrite',
        rounds: 'Rounds',
        reasoningEnabled: 'Reasoning',
        reasoningEffort: 'Thinking Effort',
        selectedTemplate: 'Template'
    };

    /** List of currently overridden fields with labels and reset actions. */
    let overrides = $derived.by(() => {
        const items: { field: keyof DatasetDefaults; label: string }[] = [];
        if (maxTokensOverridden) {
            items.push({ field: 'maxTokens', label: FIELD_LABELS.maxTokens });
        }
        if (imageQualityOverridden) {
            items.push({ field: 'imageQuality', label: FIELD_LABELS.imageQuality });
        }
        if (draftNameOverridden) {
            items.push({ field: 'draftName', label: FIELD_LABELS.draftName });
        }
        if (overwriteOverridden) {
            items.push({ field: 'overwrite', label: FIELD_LABELS.overwrite });
        }
        if (roundsOverridden) {
            items.push({ field: 'rounds', label: FIELD_LABELS.rounds });
        }
        if (reasoningEnabledOverridden) {
            items.push({ field: 'reasoningEnabled', label: FIELD_LABELS.reasoningEnabled });
        }
        if (reasoningEffortOverridden) {
            items.push({ field: 'reasoningEffort', label: FIELD_LABELS.reasoningEffort });
        }
        if (templateOverridden) {
            items.push({ field: 'selectedTemplate', label: FIELD_LABELS.selectedTemplate });
        }
        return items;
    });

    let overriddenCount = $derived(overrides.length);

    // --- Overrides section state ---
    let overridesExpanded = $state(false);

    // --- Load data on mount ---

    let dataLoaded = $state(false);
    let _pendingModelName = '';
    let _pendingApiUrl = '';

    $effect(() => {
        if (!dataLoaded) {
            dataLoaded = true;

            // Restore last-used settings from localStorage
            const saved = get(captionSettings);
            selectedEnv = saved.env;
            maxTokens = saved.maxTokens;
            imageQuality = saved.imageQuality;
            draftName = saved.draftName;
            overwrite = saved.overwrite;
            rounds = saved.rounds;
            reasoningEnabled = saved.reasoningEnabled;
            reasoningEffort = saved.reasoningEffort;
            selectedTemplate = saved.selectedTemplate;

            // Stash model for URL-guarded restore after env loads
            _pendingModelName = saved.apiModelName;
            _pendingApiUrl = saved.apiUrl;

            void _datasetName;
            void _onclose;
            loadTemplateList();
            loadDatasetDefaults();
        }
    });

    // Restore saved model once the env has loaded and the URL matches
    $effect(() => {
        if (_pendingModelName && envUrl && envUrl === _pendingApiUrl) {
            envModelName = _pendingModelName;
            _pendingModelName = '';
            _pendingApiUrl = '';
        }
    });

    // --- Dataset config loading ---

    async function loadDatasetDefaults() {
        try {
            const config = await fetchConfig(_datasetName);
            const p = config.parsed;

            datasetDefaults = {
                maxTokens: p.settings?.max_tokens ?? HARDCODED_DEFAULTS.maxTokens,
                imageQuality: p.settings?.image_quality ?? HARDCODED_DEFAULTS.imageQuality,
                draftName: HARDCODED_DEFAULTS.draftName,
                overwrite: p.overwrite_captions ?? HARDCODED_DEFAULTS.overwrite,
                rounds: p.rounds ?? HARDCODED_DEFAULTS.rounds,
                reasoningEnabled: p.reasoning?.enable ?? HARDCODED_DEFAULTS.reasoningEnabled,
                reasoningEffort: p.reasoning?.thinking_effort ?? HARDCODED_DEFAULTS.reasoningEffort,
                selectedTemplate: p.prompt?.name ?? HARDCODED_DEFAULTS.selectedTemplate
            };

            // Pre-fill fields that have no localStorage override with dataset defaults.
            // Only override if the saved value matches the hardcoded default (user never changed it).
            const saved = get(captionSettings);
            if (saved.maxTokens === HARDCODED_DEFAULTS.maxTokens) {
                maxTokens = datasetDefaults.maxTokens;
            }
            if (saved.imageQuality === HARDCODED_DEFAULTS.imageQuality) {
                imageQuality = datasetDefaults.imageQuality;
            }
            if (saved.draftName === HARDCODED_DEFAULTS.draftName) {
                draftName = datasetDefaults.draftName;
            }
            if (saved.overwrite === HARDCODED_DEFAULTS.overwrite) {
                overwrite = datasetDefaults.overwrite;
            }
            if (saved.rounds === HARDCODED_DEFAULTS.rounds) {
                rounds = datasetDefaults.rounds;
            }
            if (saved.reasoningEnabled === HARDCODED_DEFAULTS.reasoningEnabled) {
                reasoningEnabled = datasetDefaults.reasoningEnabled;
            }
            if (saved.reasoningEffort === HARDCODED_DEFAULTS.reasoningEffort) {
                reasoningEffort = datasetDefaults.reasoningEffort;
            }
            if (
                saved.selectedTemplate === HARDCODED_DEFAULTS.selectedTemplate &&
                datasetDefaults.selectedTemplate
            ) {
                selectedTemplate = datasetDefaults.selectedTemplate;
            }
        } catch {
            // Config not available — use hardcoded defaults (already set)
        }
    }

    // --- Per-field reset ---

    function resetField(field: keyof DatasetDefaults) {
        const val = datasetDefaults[field];
        switch (field) {
            case 'maxTokens':
                maxTokens = val as number;
                break;
            case 'imageQuality':
                imageQuality = val as 'auto' | 'high' | 'low';
                break;
            case 'draftName':
                draftName = val as string;
                break;
            case 'overwrite':
                overwrite = val as boolean;
                break;
            case 'rounds':
                rounds = val as number;
                break;
            case 'reasoningEnabled':
                reasoningEnabled = val as boolean;
                break;
            case 'reasoningEffort':
                reasoningEffort = val as 'low' | 'medium' | 'high';
                break;
            case 'selectedTemplate':
                selectedTemplate = val as string;
                break;
        }
    }

    function resetAllOverrides() {
        for (const field of Object.keys(HARDCODED_DEFAULTS) as (keyof DatasetDefaults)[]) {
            resetField(field);
        }
    }

    // --- Template loading & selection ---

    async function loadTemplateList() {
        isLoadingTemplates = true;
        templatesError = null;
        try {
            const list = await refreshTemplates();
            const hasDefault = list.some((t) => t.name === 'default');
            if (hasDefault && !selectedTemplate) {
                selectedTemplate = 'default';
            }
        } catch {
            templatesError = 'Failed to load templates';
        } finally {
            isLoadingTemplates = false;
        }
    }

    $effect(() => {
        const name = selectedTemplate;
        if (!name || isNewTemplate) {
            return;
        }

        let cancelled = false;
        isLoadingTemplate = true;
        (async () => {
            try {
                const info = await fetchTemplate(name);
                if (cancelled) {
                    return;
                }
                templateContent = info.content;
                templateSource = info.source;
                templateDirty = false;
            } catch {
                if (cancelled) {
                    return;
                }
                templateContent = '';
                templateSource = '';
            } finally {
                if (!cancelled) {
                    isLoadingTemplate = false;
                }
            }
        })();

        return () => {
            cancelled = true;
        };
    });

    function handleTemplateChange() {
        templateDirty = false;
        isNewTemplate = false;
        newTemplateName = '';
        templateSaveError = null;
    }

    function startNewTemplate() {
        isNewTemplate = true;
        newTemplateName = '';
        templateContent = '';
        templateSource = '';
        templateDirty = false;
        templateSaveError = null;
    }

    function cancelNewTemplate() {
        isNewTemplate = false;
        newTemplateName = '';
        templateSaveError = null;
        if (selectedTemplate) {
            const name = selectedTemplate;
            selectedTemplate = '';
            selectedTemplate = name;
        }
    }

    function handleTemplateContentChange(content: string) {
        templateContent = content;
        templateDirty = true;
    }

    async function handleSaveTemplate() {
        const name = newTemplateName.trim();
        if (!name) {
            templateSaveError = 'Template name is required';
            return;
        }

        isSavingTemplate = true;
        templateSaveError = null;
        try {
            const info = await saveTemplate(name, templateContent);
            isNewTemplate = false;
            selectedTemplate = info.name;
            templateSource = info.source;
            templateDirty = false;
            await loadTemplateList();
        } catch (e) {
            templateSaveError = friendlyErrorMessage(e, 'Failed to save template');
        } finally {
            isSavingTemplate = false;
        }
    }

    // --- Assemble options ---

    let _assembledOptions: CaptionOptions = $derived({
        env: selectedEnv,
        api_url: envUrl.trim() || undefined,
        api_token: envToken || undefined,
        api_model_name: effectiveModelName || undefined,
        prompt_template: templateDirty ? templateContent : undefined,
        prompt_name: templateDirty ? undefined : effectiveTemplateName || undefined,
        max_tokens: maxTokens,
        image_quality: imageQuality,
        rounds: rounds > 1 ? rounds : undefined,
        draft: draftName.trim() || undefined,
        overwrite: overwrite || undefined,
        reasoning: reasoningEnabled || undefined,
        reasoning_effort: reasoningEnabled ? reasoningEffort : undefined
    });

    $effect(() => {
        currentOptions = _assembledOptions;
    });

    function handleStart() {
        captionSettings.update((s) => ({
            ...s,
            env: selectedEnv,
            maxTokens,
            imageQuality,
            draftName,
            overwrite,
            rounds,
            reasoningEnabled,
            reasoningEffort,
            selectedTemplate: effectiveTemplateName || selectedTemplate,
            apiUrl: envUrl.trim(),
            apiModelName: envModelName.trim()
        }));

        promptNotificationsOnce();
        onstart?.(_assembledOptions);
    }
</script>

<div class="flex h-full flex-col">
    <!-- Scrollable content -->
    <div class="flex-1 space-y-5 overflow-y-auto p-4">
        <!-- ═══ Section: Environment ═══ -->
        <EnvSelector
            bind:env={selectedEnv}
            bind:apiUrl={envUrl}
            bind:apiToken={envToken}
            bind:apiModelName={envModelName}
        />

        <!-- ═══ Section: Template ═══ -->
        <section class="space-y-3">
            <h3 class="section-heading">Template</h3>

            {#if templatesError}
                <div class="alert-error">{templatesError}</div>
            {/if}

            <div class="flex items-end gap-2">
                <div class="flex-1">
                    {#if isNewTemplate}
                        <label class="label" for="new-tpl-name">New Template Name</label>
                        <input
                            id="new-tpl-name"
                            type="text"
                            bind:value={newTemplateName}
                            class="input"
                            placeholder="my-template"
                        />
                    {:else}
                        <div class="mb-1 flex items-center gap-1.5">
                            <label class="label mb-0" for="caption-template">Template</label>
                            {#if templateOverridden}
                                <span class="diff-dot" title="Differs from dataset config"></span>
                            {/if}
                        </div>
                        <select
                            id="caption-template"
                            class="input cursor-pointer"
                            bind:value={selectedTemplate}
                            onchange={handleTemplateChange}
                            disabled={isLoadingTemplates}
                        >
                            {#each $templates.items as t (t.name)}
                                <option value={t.name}>
                                    {t.name}{#if t.source === 'builtin'}
                                        (built-in){/if}
                                </option>
                            {/each}
                        </select>
                    {/if}
                </div>

                <div class="flex gap-2 pb-px">
                    {#if isNewTemplate}
                        <button class="btn-secondary px-3 py-2" onclick={cancelNewTemplate}>
                            Cancel
                        </button>
                        <button
                            class="btn-primary px-3 py-2"
                            onclick={handleSaveTemplate}
                            disabled={isSavingTemplate || !newTemplateName.trim()}
                        >
                            {isSavingTemplate ? 'Saving…' : 'Save'}
                        </button>
                    {:else}
                        <button
                            class="btn-secondary px-3 py-2"
                            onclick={startNewTemplate}
                            title="Create new template"
                        >
                            <SvgPlus class="h-4 w-4" />
                        </button>
                        {#if templateDirty && selectedTemplate}
                            <button
                                class="btn-primary px-3 py-2"
                                onclick={handleSaveTemplate}
                                disabled={isSavingTemplate}
                                title="Save changes to template"
                            >
                                {isSavingTemplate ? '…' : 'Save'}
                            </button>
                        {/if}
                    {/if}
                </div>
            </div>

            {#if templateSaveError}
                <p class="text-xs text-error">{templateSaveError}</p>
            {/if}

            <div class="relative">
                {#if isLoadingTemplate}
                    <SpinnerBlock class="py-8" size="h-4 w-4" label="Loading template…" />
                {:else}
                    <JinjaEditor
                        class="text-sm"
                        value={templateContent}
                        onchange={handleTemplateContentChange}
                    />
                {/if}
            </div>

            {#if templateSource === 'builtin' && !templateDirty}
                <p class="text-xs text-gray-500">
                    Built-in template. Edit above and save to create a user override.
                </p>
            {/if}
        </section>

        <!-- ═══ Section: Options ═══ -->
        <section class="space-y-3">
            <h3 class="section-heading">Options</h3>

            <div class="grid grid-cols-2 gap-3">
                <!-- Max Tokens -->
                <div>
                    <div class="mb-1 flex items-center gap-1.5">
                        <label class="label mb-0" for="caption-tokens">Max Tokens</label>
                        {#if maxTokensOverridden}
                            <span class="diff-dot" title="Differs from dataset config"></span>
                        {/if}
                    </div>
                    <input
                        id="caption-tokens"
                        type="number"
                        bind:value={maxTokens}
                        min={100}
                        max={16384}
                        class="input"
                    />
                </div>

                <!-- Image Quality -->
                <div>
                    <div class="mb-1 flex items-center gap-1.5">
                        <label class="label mb-0" for="caption-quality">Image Quality</label>
                        {#if imageQualityOverridden}
                            <span class="diff-dot" title="Differs from dataset config"></span>
                        {/if}
                    </div>
                    <select
                        id="caption-quality"
                        class="input cursor-pointer"
                        bind:value={imageQuality}
                    >
                        <option value="auto">Auto</option>
                        <option value="high">High</option>
                        <option value="low">Low</option>
                    </select>
                </div>

                <!-- Draft -->
                <div>
                    <div class="mb-1 flex items-center gap-1.5">
                        <label class="label mb-0" for="caption-draft">Draft Name</label>
                        {#if draftNameOverridden}
                            <span class="diff-dot" title="Differs from dataset config"></span>
                        {/if}
                    </div>
                    <input
                        id="caption-draft"
                        type="text"
                        bind:value={draftName}
                        class="input"
                        placeholder="(none — write caption directly)"
                    />
                </div>

                <!-- Rounds -->
                <div>
                    <div class="mb-1 flex items-center gap-1.5">
                        <label class="label mb-0" for="caption-rounds">Rounds</label>
                        {#if roundsOverridden}
                            <span class="diff-dot" title="Differs from dataset config"></span>
                        {/if}
                    </div>
                    <input
                        id="caption-rounds"
                        type="number"
                        bind:value={rounds}
                        min={1}
                        max={10}
                        class="input"
                    />
                </div>
            </div>

            <!-- Overwrite -->
            <div class="flex items-center gap-2">
                <Checkbox id="caption-overwrite" bind:checked={overwrite} />
                <label class="cursor-pointer text-sm text-gray-300" for="caption-overwrite">
                    Overwrite existing captions
                </label>
                {#if overwriteOverridden}
                    <span class="diff-dot" title="Differs from dataset config"></span>
                {/if}
            </div>
        </section>

        <!-- ═══ Section: Reasoning ═══ -->
        <section class="space-y-3">
            <div class="flex items-center gap-2">
                <Checkbox id="caption-reasoning" bind:checked={reasoningEnabled} />
                <label class="section-heading cursor-pointer" for="caption-reasoning">
                    Reasoning
                </label>
                {#if reasoningEnabledOverridden}
                    <span class="diff-dot" title="Differs from dataset config"></span>
                {/if}
            </div>

            {#if reasoningEnabled}
                <div>
                    <div class="mb-1 flex items-center gap-1.5">
                        <label class="label mb-0" for="caption-effort">Thinking Effort</label>
                        {#if reasoningEffortOverridden}
                            <span class="diff-dot" title="Differs from dataset config"></span>
                        {/if}
                    </div>
                    <select
                        id="caption-effort"
                        class="input cursor-pointer"
                        bind:value={reasoningEffort}
                    >
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                    </select>
                </div>
            {/if}
        </section>

        <!-- ═══ Section: Overrides ═══ -->
        {#if overriddenCount > 0}
            <section class="overflow-hidden rounded-lg border border-border">
                <button
                    class="flex w-full cursor-pointer items-center gap-2 px-3 py-2.5 text-sm text-gray-300 transition-colors hover:bg-surface"
                    onclick={() => (overridesExpanded = !overridesExpanded)}
                >
                    <SvgChevronLeft
                        class="h-4 w-4 transition-transform {overridesExpanded
                            ? '-rotate-90'
                            : '-rotate-180'}"
                    />
                    <span class="flex-1 text-left">Overrides</span>
                    <span class="badge-accent">{overriddenCount}</span>
                </button>

                {#if overridesExpanded}
                    <div class="space-y-1 border-t border-border px-3 py-2">
                        {#each overrides as item (item.field)}
                            <div class="flex items-center justify-between py-1.5">
                                <span class="text-sm text-gray-300">{item.label}</span>
                                <button
                                    class="cursor-pointer text-xs text-accent transition-colors hover:text-accent-hover"
                                    onclick={() => resetField(item.field)}
                                >
                                    Reset
                                </button>
                            </div>
                        {/each}

                        <div class="border-t border-border pt-1">
                            <button
                                class="w-full cursor-pointer py-1.5 text-xs text-gray-400 transition-colors hover:text-white"
                                onclick={resetAllOverrides}
                            >
                                Reset all overrides
                            </button>
                        </div>
                    </div>
                {/if}
            </section>
        {/if}
    </div>

    <!-- Footer: sticky start/stop button -->
    <div class="flex-shrink-0 border-t border-border p-4">
        {#if isBatchCaptioning}
            <button
                class="btn w-full px-4 py-2 text-sm border-error/40 bg-error/20 text-error hover:bg-error/30"
                onclick={() => {
                    onstop?.();
                    _onclose?.();
                }}
            >
                Stop Captioning
            </button>
        {:else}
            <button class="btn-primary w-full" onclick={handleStart}> Start Captioning </button>
        {/if}
    </div>
</div>
