<script lang="ts">
    import EnvSelector from '$lib/components/settings/EnvSelector.svelte';
    import CaptionOptionsFields from '$lib/components/settings/CaptionOptionsFields.svelte';
    import type { CaptionOptionsDiffDefaults } from '$lib/components/settings/CaptionOptionsFields.svelte';
    import JinjaEditor from '$lib/components/ui/JinjaEditor.svelte';
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
    import { captioningStatus } from '$lib/stores/events';
    import {
        captionOptions as captionOptionsStore,
        startBatchCaptioning,
        stopCaptioning
    } from '$lib/stores/captionActions';
    import { promptNotificationsOnce } from '$lib/notifications';
    import { fetchConfig } from '$lib/stores/configs';
    import { get } from 'svelte/store';
    import { friendlyErrorMessage, PasswordRequiredError } from '$lib/api';
    import { PasswordPromptCancelled } from '$lib/stores/passwordPrompt';
    import { toast } from '$lib/stores/toasts';

    // --- Props ---

    interface Props {
        /** Which dataset this is for. */
        datasetName: string;
        /** Called when the panel wants to close (e.g. after starting/stopping). */
        onclose?: () => void;
    }

    let { datasetName: _datasetName, onclose: _onclose }: Props = $props();

    // Derive batch captioning state from the global SSE store
    let isBatchCaptioning = $derived(
        $captioningStatus?.dataset_name === _datasetName &&
            ($captioningStatus.status === 'running' || $captioningStatus.status === 'stopping')
    );

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

    let maxTokens: number | null = $state(512);
    let imageQuality: 'auto' | 'high' | 'low' | null = $state('auto');
    let draftName = $state('');
    let overwrite = $state(false);
    let rounds: number | null = $state(1);

    // --- State: Reasoning ---

    let reasoningEnabled = $state(false);
    let reasoningEffort: 'low' | 'medium' | 'high' = $state('low');

    // --- State: General ---

    let isLoadingTemplates = $state(false);
    let templatesError: string | null = $state(null);

    // --- Dataset config defaults ---

    const HARDCODED_DEFAULTS = {
        maxTokens: 512 as number | null,
        imageQuality: 'auto' as 'auto' | 'high' | 'low' | null,
        draftName: '',
        overwrite: false,
        rounds: 1 as number | null,
        reasoningEnabled: false,
        reasoningEffort: 'low' as 'low' | 'medium' | 'high',
        storeConversation: false,
        reasoningExcludeOutput: true,
        selectedTemplate: ''
    };

    let datasetDefaults: CaptionOptionsDiffDefaults & { selectedTemplate: string } = $state({
        ...HARDCODED_DEFAULTS
    });

    // --- Computed ---

    let effectiveModelName = $derived(envModelName.trim());
    let templateDirty = $state(false);
    let effectiveTemplateName = $derived(isNewTemplate ? newTemplateName.trim() : selectedTemplate);

    // --- Diff tracking (for overrides section) ---

    type OverrideableField = keyof CaptionOptionsDiffDefaults | 'selectedTemplate';

    const FIELD_LABELS: Record<OverrideableField, string> = {
        maxTokens: 'Max Tokens',
        imageQuality: 'Image Quality',
        draftName: 'Draft Name',
        overwrite: 'Overwrite',
        rounds: 'Rounds',
        reasoningEnabled: 'Reasoning',
        reasoningEffort: 'Thinking Effort',
        storeConversation: 'Store Conversation',
        reasoningExcludeOutput: 'Exclude Reasoning',
        selectedTemplate: 'Template'
    };

    let templateOverridden = $derived(selectedTemplate !== datasetDefaults.selectedTemplate);

    /** List of currently overridden fields with labels and reset actions. */
    let overrides = $derived.by(() => {
        const items: { field: OverrideableField; label: string }[] = [];
        // Options fields
        if (maxTokens !== datasetDefaults.maxTokens) {
            items.push({ field: 'maxTokens', label: FIELD_LABELS.maxTokens });
        }
        if (imageQuality !== datasetDefaults.imageQuality) {
            items.push({ field: 'imageQuality', label: FIELD_LABELS.imageQuality });
        }
        if (draftName !== datasetDefaults.draftName) {
            items.push({ field: 'draftName', label: FIELD_LABELS.draftName });
        }
        if (overwrite !== datasetDefaults.overwrite) {
            items.push({ field: 'overwrite', label: FIELD_LABELS.overwrite });
        }
        if (rounds !== datasetDefaults.rounds) {
            items.push({ field: 'rounds', label: FIELD_LABELS.rounds });
        }
        if (reasoningEnabled !== datasetDefaults.reasoningEnabled) {
            items.push({ field: 'reasoningEnabled', label: FIELD_LABELS.reasoningEnabled });
        }
        if (reasoningEnabled && reasoningEffort !== datasetDefaults.reasoningEffort) {
            items.push({ field: 'reasoningEffort', label: FIELD_LABELS.reasoningEffort });
        }
        // Template (managed here, not in CaptionOptionsFields)
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
                draftName: '',
                overwrite: p.overwrite_captions ?? HARDCODED_DEFAULTS.overwrite,
                rounds: p.rounds ?? HARDCODED_DEFAULTS.rounds,
                reasoningEnabled: p.reasoning?.enable ?? HARDCODED_DEFAULTS.reasoningEnabled,
                reasoningEffort: p.reasoning?.thinking_effort ?? HARDCODED_DEFAULTS.reasoningEffort,
                storeConversation: false,
                reasoningExcludeOutput: true,
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

    function resetField(field: OverrideableField) {
        const val = datasetDefaults[field];
        switch (field) {
            case 'maxTokens':
                maxTokens = val as number | null;
                break;
            case 'imageQuality':
                imageQuality = val as 'auto' | 'high' | 'low' | null;
                break;
            case 'draftName':
                draftName = val as string;
                break;
            case 'overwrite':
                overwrite = val as boolean;
                break;
            case 'rounds':
                rounds = val as number | null;
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
        for (const field of Object.keys(HARDCODED_DEFAULTS) as OverrideableField[]) {
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

    function handleTemplateContentChange() {
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
        max_tokens: maxTokens ?? undefined,
        image_quality: imageQuality ?? undefined,
        rounds: rounds && rounds > 1 ? rounds : undefined,
        draft: draftName.trim() || undefined,
        overwrite: overwrite || undefined,
        reasoning: reasoningEnabled || undefined,
        reasoning_effort: reasoningEnabled ? reasoningEffort : undefined
    });

    // Keep the shared store in sync with assembled options so
    // single-image captioning from ImageDetail always uses current settings.
    $effect(() => {
        captionOptionsStore.set(_assembledOptions);
    });

    async function handleStart() {
        captionSettings.update((s) => ({
            ...s,
            env: selectedEnv,
            maxTokens: maxTokens ?? 512,
            imageQuality: imageQuality ?? 'auto',
            draftName,
            overwrite,
            rounds: rounds ?? 1,
            reasoningEnabled,
            reasoningEffort,
            selectedTemplate: effectiveTemplateName || selectedTemplate,
            apiUrl: envUrl.trim(),
            apiModelName: envModelName.trim()
        }));

        promptNotificationsOnce();
        try {
            await startBatchCaptioning(_datasetName);
        } catch (e) {
            if (e instanceof PasswordPromptCancelled) {
                return;
            }
            if (e instanceof PasswordRequiredError) {
                toast.error('Incorrect password. Please try again.');
                return;
            }
            toast.error(friendlyErrorMessage(e, 'Failed to start captioning'));
            return;
        }
        _onclose?.();
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

        <!-- ═══ Section: Options & Reasoning (shared component) ═══ -->
        <CaptionOptionsFields
            bind:maxTokens
            bind:imageQuality
            bind:rounds
            bind:overwrite
            bind:reasoningEnabled
            bind:reasoningEffort
            bind:draftName
            display={{
                idPrefix: 'caption',
                showDraftName: true,
                diffDefaults: datasetDefaults
            }}
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

            <div class="relative h-64">
                {#if isLoadingTemplate}
                    <SpinnerBlock class="py-8" size="h-4 w-4" label="Loading template…" />
                {:else}
                    <JinjaEditor
                        class="rounded-md border border-border text-sm h-full"
                        bind:value={templateContent}
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
                class="btn w-full border-error/40 bg-error/20 px-4 py-2 text-sm text-error hover:bg-error/30"
                onclick={() => {
                    stopCaptioning(_datasetName);
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
