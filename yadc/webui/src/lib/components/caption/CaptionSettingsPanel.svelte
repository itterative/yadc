<script lang="ts">
    import EnvSelector from '$lib/components/env/EnvSelector.svelte';
    import CaptionOptionsFields from '$lib/components/settings/CaptionOptionsFields.svelte';
    import type { CaptionOptionsDiffDefaults } from '$lib/components/settings/CaptionOptionsFields.svelte';
    import {
        captionOptions as captionOptionsStore,
        startBatchCaptioning,
        stopCaptioning
    } from '$lib/stores/caption';
    import type { CaptionOptions } from '$lib/stores/caption';
    import { captionSettings } from '$lib/stores/caption';
    import { captioningStatus } from '$lib/stores/events';
    import { promptNotificationsOnce } from '$lib/notifications';
    import { deferred } from '$lib/async';
    import { fetchConfig } from '$lib/stores/config';
    import { get } from 'svelte/store';
    import { friendlyErrorMessage, PasswordRequiredError } from '$lib/api';
    import { PasswordPromptCancelled } from '$lib/stores/passwordPrompt';
    import { toast } from '$lib/stores/toasts';
    import TemplateSection from './TemplateSection.svelte';
    import OverridesSection, { type OverrideItem } from './OverridesSection.svelte';

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
            ($captioningStatus.status === 'starting' ||
                $captioningStatus.status === 'running' ||
                $captioningStatus.status === 'stopping')
    );

    // --- State: Environment (managed by EnvSelector via bindings) ---

    let selectedEnv = $state('default');
    let envUrl = $state('');
    let envToken = $state('');
    let envModelName = $state('');

    // --- State: Template (managed by TemplateSection via bindings) ---

    let selectedTemplate = $state('');

    // --- State: Options ---

    let maxTokens: number | null = $state(512);
    let imageQuality: 'auto' | 'high' | 'low' | null = $state('auto');
    let draftName = $state('');
    let overwrite = $state(false);
    let rounds: number | null = $state(1);
    let batchSize: number | null = $state(1);

    // --- State: Reasoning ---

    let reasoningEnabled = $state(false);
    let reasoningEffort: 'low' | 'medium' | 'high' = $state('low');

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
    let effectiveTemplateName = $derived(selectedTemplate);

    let templateOverridden = $derived(selectedTemplate !== datasetDefaults.selectedTemplate);

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

    /** List of currently overridden fields with labels and reset actions. */
    let overrides: OverrideItem[] = $derived.by(() => {
        const items: OverrideItem[] = [];
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
        // Template (managed in TemplateSection, but the override check stays here
        // because the rest of the diff tracking lives in the host).
        if (templateOverridden) {
            items.push({ field: 'selectedTemplate', label: FIELD_LABELS.selectedTemplate });
        }
        return items;
    });

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
            batchSize = saved.batchSize;
            reasoningEnabled = saved.reasoningEnabled;
            reasoningEffort = saved.reasoningEffort;
            selectedTemplate = saved.selectedTemplate;

            // Stash model for URL-guarded restore after env loads
            _pendingModelName = saved.apiModelName;
            _pendingApiUrl = saved.apiUrl;

            void _datasetName;
            void _onclose;
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

    function resetField(field: string) {
        switch (field) {
            case 'maxTokens':
                maxTokens = datasetDefaults.maxTokens;
                break;
            case 'imageQuality':
                imageQuality = datasetDefaults.imageQuality;
                break;
            case 'draftName':
                draftName = datasetDefaults.draftName;
                break;
            case 'overwrite':
                overwrite = datasetDefaults.overwrite;
                break;
            case 'rounds':
                rounds = datasetDefaults.rounds;
                break;
            case 'reasoningEnabled':
                reasoningEnabled = datasetDefaults.reasoningEnabled;
                break;
            case 'reasoningEffort':
                reasoningEffort = datasetDefaults.reasoningEffort;
                break;
            case 'selectedTemplate':
                selectedTemplate = datasetDefaults.selectedTemplate;
                break;
            default:
                // Unknown field — ignore (forward-compat with new diff types).
                break;
        }
    }

    function resetAllOverrides() {
        for (const field of Object.keys(HARDCODED_DEFAULTS)) {
            resetField(field);
        }
    }

    // --- Assemble options ---

    let _assembledOptions: CaptionOptions = $derived({
        env: selectedEnv,
        api_url: envUrl.trim() || undefined,
        api_token: envToken || undefined,
        api_model_name: effectiveModelName || undefined,
        prompt_name: effectiveTemplateName || undefined,
        max_tokens: maxTokens ?? undefined,
        image_quality: imageQuality ?? undefined,
        rounds: rounds && rounds > 1 ? rounds : undefined,
        draft: draftName.trim() || undefined,
        overwrite: overwrite || undefined,
        max_concurrent: batchSize && batchSize > 1 ? batchSize : undefined,
        reasoning: reasoningEnabled || undefined,
        reasoning_effort: reasoningEnabled ? reasoningEffort : undefined
    });

    // Keep the shared store in sync with assembled options so
    // single-image captioning from ImageDetail always uses current settings.
    $effect(() => {
        captionOptionsStore.set(_assembledOptions);
    });

    function _buildSettings(): import('$lib/stores/caption').CaptionSettings {
        return {
            $version: 2,
            env: selectedEnv,
            maxTokens: maxTokens ?? 512,
            imageQuality: imageQuality ?? 'auto',
            draftName,
            overwrite,
            rounds: rounds ?? 1,
            batchSize: batchSize ?? 1,
            reasoningEnabled,
            reasoningEffort,
            selectedTemplate: effectiveTemplateName || selectedTemplate,
            apiUrl: envUrl.trim(),
            apiModelName: envModelName.trim()
        };
    }

    const persistSettings = deferred(() => {
        captionSettings.set(_buildSettings());
    }, 300);

    $effect(() => {
        if (!dataLoaded) {
            return;
        }
        // Track every field we want to persist so the effect re-runs on change.
        void selectedEnv;
        void maxTokens;
        void imageQuality;
        void draftName;
        void overwrite;
        void rounds;
        void batchSize;
        void reasoningEnabled;
        void reasoningEffort;
        void selectedTemplate;
        void effectiveTemplateName;
        void envUrl;
        void envModelName;

        persistSettings();
    });

    async function handleStart() {
        // Ensure any pending debounced save is flushed before starting.
        persistSettings();

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

        <!-- ═══ Section: Concurrency (caption-flow only) ═══ -->
        <section class="space-y-2">
            <h3 class="section-heading">Concurrency</h3>
            <div class="max-w-xs">
                <label class="label mb-1 block" for="caption-batch-size">
                    Concurrent requests
                </label>
                <input
                    id="caption-batch-size"
                    type="number"
                    min={1}
                    max={32}
                    value={batchSize ?? ''}
                    class="input"
                    oninput={(e) => {
                        const raw = (e.target as HTMLInputElement).value;
                        batchSize = raw === '' ? 1 : Number(raw);
                    }}
                />
                <p class="help-text">
                    How many captioning requests can be in flight at once. 1 = sequential (default).
                    Higher values trade rate-limit risk for throughput — start low and increase for
                    local backends (vLLM, llama.cpp) that can handle parallel load.
                </p>
            </div>
        </section>

        <!-- ═══ Section: Template ═══ -->
        <TemplateSection
            bind:selectedTemplate
            datasetDefaultTemplate={datasetDefaults.selectedTemplate}
        />

        <!-- ═══ Section: Overrides ═══ -->
        <OverridesSection
            {overrides}
            bind:expanded={overridesExpanded}
            onReset={resetField}
            onResetAll={resetAllOverrides}
        />
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
