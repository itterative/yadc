<script lang="ts">
    import SvgSave from '$lib/icons/SvgSave.svelte';
    import SvgSparkle from '$lib/icons/SvgSparkle.svelte';
    import PillTabs from '$lib/components/ui/tabs/PillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import {
        startGeneration,
        generation,
        promptSettings,
        saveHistoryEntry
    } from '$lib/stores/prompts';
    import type {
        ExamplePair,
        PromptGenFocus,
        PromptGenMode,
        PromptHistoryEntry,
        PromptImageQuality
    } from '$lib/stores/prompts';
    import { friendlyErrorMessage } from '$lib/api';
    import { toast } from '$lib/stores/toasts';
    import { EditTemplateDialog } from '$lib/components/templates';
    import PromptForm from './PromptForm.svelte';
    import PromptSettings from './PromptSettings.svelte';
    import GenerationPreview from './GenerationPreview.svelte';
    import PromptHistoryPanel from './PromptHistoryPanel.svelte';

    // Three tabs: Generate (the single form, hosting New/Refine via
    // the inline mode pills), Settings (env/model + generation
    // limits), and History. Each tab holds genuinely different
    // content. ``mode`` is the form's mode (persisted, matches
    // PromptGenMode), driven by the form's pills through
    // ``bind:mode``. ``activeTab`` only switches between the three
    // panels.
    type ActiveTab = 'generate' | 'settings' | 'history';

    // Few-shot examples live in memory only — the history is a
    // server-side persistent store (Phase 7), so saving and
    // restoring the examples is a one-click operation rather than
    // relying on localStorage / IndexedDB.
    let examples = $state<ExamplePair[]>([]);

    // Controlled-component pattern: local ``$state`` is the source of
    // truth for the form, and a single ``$effect`` mirrors every
    // change into the persisted ``promptSettings`` store. We avoid
    // ``bind:prop={$store.field}`` because Svelte 5's ``$store`` is a
    // read-only auto-subscription — writes to ``$store.field`` don't
    // propagate back to the store, which leads to the form being
    // silently disconnected from persistence (and can produce effect
    // loops in downstream consumers that watch both sides). Mirrors
    // the pattern used by ``GeneralSettings.svelte`` for
    // ``yadc/settings`` (one-way: state → store).
    const _initial = $promptSettings;
    let mode = $state<PromptGenMode>(_initial.mode);
    let env = $state(_initial.env);
    let apiUrl = $state(_initial.apiUrl);
    let apiToken = $state(_initial.apiToken);
    let apiModelName = $state(_initial.apiModelName);
    let intent = $state(_initial.intent);
    let focus = $state<PromptGenFocus>(_initial.focus);
    let maxTokens = $state(_initial.maxTokens);
    let imageQuality = $state<PromptImageQuality>(_initial.imageQuality);

    // Refine-mode-only. NOT persisted — on reload the user re-selects
    // a template (or pastes their own). The picker selection is also
    // ephemeral and lives inside the form.
    let templateContent = $state('');

    // Always land on the form; History is opt-in via its tab.
    let activeTab: ActiveTab = $state<ActiveTab>('generate');

    $effect(() => {
        promptSettings.update((s) => ({
            ...s,
            mode,
            env,
            apiUrl,
            apiToken,
            apiModelName,
            intent,
            focus,
            maxTokens,
            imageQuality
        }));
    });

    let isStreaming = $derived(generation.status === 'streaming');
    let canGenerate = $derived.by(() => {
        if (isStreaming || env.trim().length === 0 || intent.trim().length === 0) {
            return false;
        }
        // In refine mode the model needs a template to refine — same
        // check the backend ``PromptGenerationRequest`` validator
        // does, surfaced client-side to avoid the round-trip.
        if (mode === 'refine' && templateContent.trim().length === 0) {
            return false;
        }
        return true;
    });

    let canSave = $derived(
        intent.trim().length > 0 && (mode === 'generate' || templateContent.trim().length > 0)
    );

    let saveDialogOpen = $state(false);
    let saveBody = $state('');

    let historyPanel: PromptHistoryPanel | undefined = $state();

    function handleChange(field: string, value: unknown) {
        // Hook for future side-effects. Persistence is handled by the
        // ``$effect`` above.
        void field;
        void value;
    }

    async function handleGenerate() {
        if (!canGenerate) {
            return;
        }
        // Snapshot form values so the streaming run uses them
        // even if the user edits them mid-stream.
        await startGeneration({
            env,
            intent,
            examples: examples.map((e) => ({ ...e })),
            focus,
            apiModelName: apiModelName || null,
            maxTokens,
            imageQuality,
            templateContent: mode === 'refine' ? templateContent : null
        });
    }

    function handleSaveTemplate(body: string) {
        saveBody = body;
        saveDialogOpen = true;
    }

    function handleTemplateSaved(name: string) {
        void name;
        saveDialogOpen = false;
    }

    let isSaving = $state(false);

    async function handleSaveToHistory() {
        if (!canSave || isSaving) {
            return;
        }
        isSaving = true;
        try {
            await saveHistoryEntry({
                mode,
                intent,
                focus,
                examples: examples.map((e) => ({ ...e })),
                template_content: mode === 'refine' ? templateContent : null
            });
            toast.success('Saved to history');
            // Refresh the history panel so the new entry shows up
            // when the user switches to the History tab.
            await historyPanel?.refresh();
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to save to history'));
        } finally {
            isSaving = false;
        }
    }

    function handleRestore(entry: PromptHistoryEntry) {
        // Switch to the form and apply the entry's mode + fields.
        activeTab = 'generate';
        mode = entry.mode;
        intent = entry.intent;
        focus = entry.focus;
        examples = entry.examples.map((e) => ({ ...e }));
        templateContent = entry.template_content ?? '';
        toast.success('Restored from history');
    }
</script>

<div class="grid h-full min-h-0 grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
    <!-- Form (left) — three tabs with genuinely different content:
         Generate (the prompt artifact form, hosting New/Refine via
         the inline mode pills), Settings (env/model + generation
         limits), and History. The form, settings, and examples /
         template state all live in the host as bound ``$state``, so
         switching tabs never loses anything. -->
    <div class="card flex min-h-0 flex-col overflow-hidden">
        <PillTabs bind:value={activeTab} class="min-h-0 flex-1">
            <Tab id="generate" label="Generate" class="h-full overflow-y-auto">
                <div class="p-4">
                    <PromptForm
                        bind:mode
                        bind:intent
                        bind:focus
                        bind:examples
                        bind:templateContent
                        onchange={handleChange}
                    />
                </div>
            </Tab>
            <Tab id="settings" label="Settings" class="h-full overflow-y-auto">
                <div class="p-4">
                    <PromptSettings
                        bind:env
                        bind:apiUrl
                        bind:apiToken
                        bind:apiModelName
                        bind:maxTokens
                        bind:imageQuality
                        onchange={handleChange}
                    />
                </div>
            </Tab>
            <Tab id="history" label="History" class="h-full overflow-y-auto">
                <PromptHistoryPanel bind:this={historyPanel} onrestore={handleRestore} />
            </Tab>
        </PillTabs>
        <div
            class="flex items-center justify-between border-t border-border bg-surface/50 px-4 py-2"
        >
            <p class="text-xs text-gray-500">
                {#if activeTab === 'history'}
                    {examples.length} example{examples.length === 1 ? '' : 's'} ready to save
                {:else if activeTab === 'settings'}
                    Environment &amp; generation limits
                {:else if examples.length > 0}
                    {examples.length} example{examples.length === 1 ? '' : 's'}
                {:else}
                    No examples — model will infer style from intent alone
                {/if}
            </p>
            <div class="flex items-center gap-2">
                {#if activeTab === 'generate'}
                    <button
                        class="btn-secondary flex cursor-pointer items-center gap-1.5 text-sm disabled:cursor-not-allowed disabled:opacity-50"
                        onclick={handleSaveToHistory}
                        disabled={!canSave || isSaving}
                        title="Save current prompt to history"
                    >
                        <SvgSave class="h-3.5 w-3.5" />
                        Save to history
                    </button>
                    <button
                        class="btn-primary flex cursor-pointer items-center gap-1.5 disabled:cursor-not-allowed disabled:opacity-50"
                        onclick={handleGenerate}
                        disabled={!canGenerate}
                    >
                        <SvgSparkle class="h-4 w-4" />
                        {isStreaming ? 'Streaming…' : mode === 'refine' ? 'Refine' : 'Generate'}
                    </button>
                {/if}
            </div>
        </div>
    </div>

    <!-- Preview (right) -->
    <div class="min-h-0">
        <GenerationPreview onsavetemplate={handleSaveTemplate} />
    </div>
</div>

<EditTemplateDialog
    open={saveDialogOpen}
    templateName={null}
    initialContent={saveBody}
    onclose={() => (saveDialogOpen = false)}
    onsaved={handleTemplateSaved}
/>
