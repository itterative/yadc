<script lang="ts">
    import SvgSave from '$lib/icons/SvgSave.svelte';
    import SvgSparkle from '$lib/icons/SvgSparkle.svelte';
    import { untrack } from 'svelte';
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
        PromptHistoryEntry
    } from '$lib/stores/prompts';
    import { friendlyErrorMessage } from '$lib/api';
    import { toast } from '$lib/stores/toasts';
    import { EditTemplateDialog } from '$lib/components/templates';
    import PromptForm from './PromptForm.svelte';
    import GenerationPreview from './GenerationPreview.svelte';
    import PromptHistoryPanel from './PromptHistoryPanel.svelte';

    // The PillTabs have three children: Generate, Refine, History.
    // ``mode`` is the form's mode (persisted, matches PromptGenMode);
    // ``activeTab`` is the tab the user is currently viewing. When
    // ``activeTab`` is 'history', the form is hidden and the history
    // panel is shown. ``mode`` is derived from ``activeTab`` for
    // the form tabs and left untouched when ``activeTab`` is
    // 'history' (so a restore in history mode can set mode to the
    // entry's mode without losing the user's last form mode).
    type ActiveTab = 'generate' | 'refine' | 'history';

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

    // Refine-mode-only. NOT persisted — on reload the user re-selects
    // a template (or pastes their own). The picker selection is also
    // ephemeral and lives inside the form.
    let templateContent = $state('');

    // Start on the same tab as the persisted mode (so a returning
    // user lands on Generate/Refine, never History). ``untrack``
    // makes the read one-shot — we don't want a runtime effect to
    // reset ``activeTab`` every time the user clicks a tab.
    let activeTab: ActiveTab = $state<ActiveTab>(untrack(() => mode));

    $effect(() => {
        promptSettings.update((s) => ({
            ...s,
            mode,
            env,
            apiUrl,
            apiToken,
            apiModelName,
            intent,
            focus
        }));
    });

    // When the user picks a different form tab, sync ``mode`` so the
    // form / button label / canGenerate check all agree. History tab
    // doesn't touch ``mode`` (it stays at whatever the user left it).
    $effect(() => {
        if (activeTab === 'generate' || activeTab === 'refine') {
            mode = activeTab;
        }
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
        // Switch the tab + form mode to match the entry. The order
        // matters: setting ``activeTab`` first triggers the
        // ``$effect`` that syncs ``mode``, which the form reads on
        // its next render. Then we populate the form fields.
        activeTab = entry.mode;
        mode = entry.mode;
        intent = entry.intent;
        focus = entry.focus;
        examples = entry.examples.map((e) => ({ ...e }));
        templateContent = entry.template_content ?? '';
        toast.success('Restored from history');
    }
</script>

<div class="grid h-full min-h-0 grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
    <!-- Form (left) — PillTabs host. The form is rendered in BOTH
         tab children (same instance-via-bindings) so the user can
         switch modes without losing their env / model / intent /
         examples state. The forms' internal state (loading flags,
         template picker selection, etc.) is per-instance — switching
         to the other tab re-mounts the form and its effects re-run
         (model/template fetches are debounced on the store side, so
         the double-fetch is a no-op in practice). See the Phase 6b
         history entry for the full reasoning. -->
    <div class="card flex min-h-0 flex-col overflow-hidden">
        <PillTabs bind:value={activeTab} class="min-h-0 flex-1">
            <Tab id="generate" label="Generate" class="h-full overflow-y-auto">
                <div class="p-4">
                    <PromptForm
                        mode="generate"
                        bind:env
                        bind:apiUrl
                        bind:apiToken
                        bind:apiModelName
                        bind:intent
                        bind:focus
                        bind:examples
                        bind:templateContent
                        onchange={handleChange}
                    />
                </div>
            </Tab>
            <Tab id="refine" label="Refine" class="h-full overflow-y-auto">
                <div class="p-4">
                    <PromptForm
                        mode="refine"
                        bind:env
                        bind:apiUrl
                        bind:apiToken
                        bind:apiModelName
                        bind:intent
                        bind:focus
                        bind:examples
                        bind:templateContent
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
                {:else if examples.length > 0}
                    {examples.length} example{examples.length === 1 ? '' : 's'}
                {:else}
                    No examples — model will infer style from intent alone
                {/if}
            </p>
            <div class="flex items-center gap-2">
                {#if activeTab !== 'history'}
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
