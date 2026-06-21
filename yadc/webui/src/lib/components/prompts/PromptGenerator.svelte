<script lang="ts">
    import SvgSparkle from '$lib/icons/SvgSparkle.svelte';
    import PillTabs from '$lib/components/ui/tabs/PillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import { startGeneration, generation, promptSettings } from '$lib/stores/prompts';
    import type { ExamplePair, PromptGenFocus, PromptGenMode } from '$lib/stores/prompts';
    import { EditTemplateDialog } from '$lib/components/templates';
    import PromptForm from './PromptForm.svelte';
    import GenerationPreview from './GenerationPreview.svelte';

    // Few-shot examples live in memory only — their ``image_data_url``
    // payloads are too large for localStorage and we don't track a
    // re-fetchable source per example. (See the prompt-generator plan
    // history for the deferred IndexedDB follow-up.)
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

    let saveDialogOpen = $state(false);
    let saveBody = $state('');

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
        <PillTabs bind:value={mode} class="min-h-0 flex-1">
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
        </PillTabs>
        <div
            class="flex items-center justify-between border-t border-border bg-surface/50 px-4 py-2"
        >
            <p class="text-xs text-gray-500">
                {#if examples.length > 0}
                    {examples.length} example{examples.length === 1 ? '' : 's'}
                {:else}
                    No examples — model will infer style from intent alone
                {/if}
            </p>
            <button
                class="btn-primary flex cursor-pointer items-center gap-1.5 disabled:cursor-not-allowed disabled:opacity-50"
                onclick={handleGenerate}
                disabled={!canGenerate}
            >
                <SvgSparkle class="h-4 w-4" />
                {isStreaming ? 'Streaming…' : mode === 'refine' ? 'Refine' : 'Generate'}
            </button>
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
