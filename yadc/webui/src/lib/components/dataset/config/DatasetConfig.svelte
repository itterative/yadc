<script lang="ts">
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgFile from '$lib/icons/SvgFile.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgHistory from '$lib/icons/SvgHistory.svelte';
    import CompactPillTabs from '$lib/components/ui/tabs/CompactPillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import {
        fetchConfig,
        patchConfig,
        updateConfig,
        type ConfigValidationError
    } from '$lib/stores/config';
    import { toast } from '$lib/stores/toasts';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, linkedController } from '$lib/abort';
    import { refreshTemplates } from '$lib/stores/templates';
    import { buildPatch, createPreviewScheduler } from './patch';
    import { configState } from './state.svelte';
    import ConfigHistory from './ConfigHistory.svelte';
    import DatasetConfigAdvanced from './DatasetConfigAdvanced.svelte';
    import DatasetConfigForm from './DatasetConfigForm.svelte';

    // --- Props ---

    interface Props {
        /** Which dataset this is for. */
        datasetName: string;
        /** Dataset source — managed (upload) vs external (import/create). */
        source?: 'upload' | 'import' | 'create';
        /** Called after the config is saved and rescanned. */
        onsaved?: () => void;
    }

    let { datasetName, source, onsaved }: Props = $props();

    // Abort context — children (ConfigHistory, ConfigApiSection) inherit.
    const abort = createAbortContext();

    // --- Container-only state (data fetching + UI) ---

    let isLoading = $state(false);
    let isSaving = $state(false);
    let error: string | null = $state(null);
    let validationErrors: ConfigValidationError[] = $state([]);
    let loadedConfigPath: string = $state('');

    // --- View mode ---

    let activeView: 'simplified' | 'advanced' | 'history' = $state('simplified');

    // --- Live preview ---

    let previewContent = $state('');
    const preview = createPreviewScheduler(
        () => datasetName,
        (content) => {
            previewContent = content;
        }
    );

    // --- Dirty tracking: pick the right one for the active view ---

    let dirty = $derived(
        activeView === 'simplified'
            ? configState.simplifiedDirty
            : activeView === 'advanced'
              ? configState.rawDirty
              : false
    );

    // --- View mode switching: sync raw content from preview when not dirty ---

    let previousView: 'simplified' | 'advanced' | 'history' = $state('simplified');

    $effect(() => {
        if (activeView === previousView) {
            return;
        }
        if (activeView === 'advanced' && !configState.rawDirty) {
            // Only sync preview content if there are no unsaved advanced changes.
            // Preserves in-progress raw edits when switching back from form/history.
            configState.rawContent = previewContent;
            configState.loadedRawContent = previewContent;
        }
        previousView = activeView;
    });

    // --- Load on mount / dataset change ---

    $effect(() => {
        const name = datasetName;
        if (!name) {
            return;
        }
        const controller = linkedController(abort.signal);
        loadConfig(controller.signal);
        return () => controller.abort();
    });

    async function loadConfig(signal?: AbortSignal) {
        isLoading = true;
        error = null;

        try {
            const [config] = await Promise.all([
                fetchConfig(datasetName, signal),
                refreshTemplates(signal)
            ]);
            if (signal?.aborted) {
                return;
            }
            validationErrors = config.validation_error ?? [];
            previewContent = config.content;
            loadedConfigPath = config.config_path;
            configState.populateFields(config.parsed, config.content);
        } catch (e) {
            if (signal?.aborted) {
                return;
            }
            error = friendlyErrorMessage(e, 'Failed to load config');
        } finally {
            if (!signal?.aborted) {
                isLoading = false;
            }
        }
    }

    // --- Trigger live preview on any field change ---

    $effect(() => {
        // Touch all reactive values so Svelte tracks them
        void configState.apiUrl;
        void configState.apiModelName;
        void configState.promptName;
        void configState.maxTokens;
        void configState.imageQuality;
        void configState.rounds;
        void configState.overwrite;
        void configState.storeConversation;
        void configState.reasoningEnabled;
        void configState.reasoningEffort;
        void configState.reasoningExcludeOutput;
        void configState.envName;
        void datasetName;
        // Touch dataset entries (array + each entry's fields)
        void configState.datasetEntries;
        for (const entry of configState.datasetEntries) {
            void entry.path;
            for (const e of entry.extras) {
                void e.key;
                void e.value;
            }
        }

        if (!isLoading && !error) {
            preview.schedule();
        }
    });

    // --- Save (PATCH for simplified, PUT for advanced) ---

    let configVersion = $state(0);

    function handleConfigSaved() {
        configVersion++;
        toast.success('Config saved');
        onsaved?.();
    }

    function handleHistoryRestored() {
        configVersion++;
        toast.success('Config restored');
        // Re-fetch config so the form reflects the restored on-disk state
        loadConfig();
        onsaved?.();
    }

    async function handleSave() {
        if (activeView === 'history') {
            return;
        }
        isSaving = true;

        try {
            if (activeView === 'advanced') {
                // PUT: full content replacement
                const result = await updateConfig(datasetName, configState.rawContent);
                previewContent = result.content;
                configState.rawContent = configState.loadedRawContent = result.content;
                configState.populateFields(result.parsed, result.content);
                validationErrors = result.validation_error ?? [];
            } else {
                // PATCH: merge structured fields
                const result = await patchConfig(datasetName, buildPatch());
                previewContent = result.content;
                configState.rawContent = configState.loadedRawContent = result.content;
                validationErrors = result.validation_error ?? [];
                configState.resetSimplifiedDirty();
            }

            handleConfigSaved();
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to save config'));
        } finally {
            isSaving = false;
        }
    }
</script>

<div class="flex h-full flex-col">
    <!-- Scrollable content -->
    <div class="flex-1 space-y-5 overflow-y-auto p-4">
        {#if isLoading}
            <SpinnerBlock class="py-8" size="h-5 w-5" label="Loading config…" />
        {:else if error}
            <div class="alert-error">{error}</div>
        {:else}
            {#if validationErrors.length > 0}
                <div class="alert-warning">
                    <p class="font-medium">Config validation issues:</p>
                    <ul class="mt-1 list-inside list-disc">
                        {#each validationErrors as err (err.loc.join('.'))}
                            <li>{err.loc.join('.')}: {err.msg}</li>
                        {/each}
                    </ul>
                </div>
            {/if}

            <!-- ═══ View mode toggle ═══ -->
            <CompactPillTabs storageId="dataset/config" bind:value={activeView} class="h-full">
                <Tab id="simplified" label="Form" icon={SvgFile} class="flex flex-col gap-4">
                    <DatasetConfigForm {source} {datasetName} {loadedConfigPath} {previewContent} />
                </Tab>
                <Tab id="advanced" label="Advanced" icon={SvgEdit} class="h-full">
                    <DatasetConfigAdvanced {source} />
                </Tab>
                <Tab id="history" label="History" icon={SvgHistory} class="h-full">
                    <ConfigHistory {datasetName} {configVersion} onsaved={handleHistoryRestored} />
                </Tab>
            </CompactPillTabs>
        {/if}
    </div>

    <!-- Footer: sticky save button -->
    {#if !isLoading && !error && activeView !== 'history'}
        <div class="shrink-0 border-t border-border p-4">
            <div class="flex gap-2">
                <button
                    class="btn-secondary flex-1"
                    onclick={() => loadConfig()}
                    disabled={isSaving}
                >
                    <SvgRefresh class="mr-1 inline-block h-4 w-4" />
                    Reload
                </button>
                <button
                    class="btn-primary flex-1"
                    onclick={handleSave}
                    disabled={isSaving || !dirty}
                >
                    {isSaving ? 'Saving…' : 'Save Config'}
                </button>
            </div>
        </div>
    {/if}
</div>
