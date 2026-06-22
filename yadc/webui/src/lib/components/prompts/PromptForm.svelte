<script lang="ts">
    import { templates, refreshTemplates, fetchTemplate } from '$lib/stores/templates';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, getAbortContext, linkedController } from '$lib/abort';
    import { generation } from '$lib/stores/prompts';
    import type { ExamplePair, PromptGenFocus, PromptGenMode } from '$lib/stores/prompts';
    import JinjaEditor from '$lib/components/ui/JinjaEditor.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import Card from '$lib/components/ui/Card.svelte';
    import ActionBar from '$lib/components/ui/ActionBar.svelte';
    import ActionBarItem from '$lib/components/ui/ActionBarItem.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import { EditTemplateDialog } from '$lib/components/templates';
    import ExamplesPanel from './ExamplesPanel.svelte';

    interface Props {
        /** Generate vs refine mode. Bindable — the inline New/Refine
         *  pills update it; the host persists it. */
        mode: PromptGenMode;
        intent: string;
        focus: PromptGenFocus;
        examples: ExamplePair[];
        /** Existing template body for refine mode. Bindable so the
         *  host can snapshot it for ``startGeneration``. Ephemeral —
         *  not persisted; the host re-seeds it on reload via the
         *  template picker. */
        templateContent: string;
        onchange: (field: string, value: unknown) => void;
    }

    let {
        mode = $bindable('generate'),
        intent = $bindable(''),
        focus = $bindable('both'),
        examples = $bindable([]),
        templateContent = $bindable(''),
        onchange
    }: Props = $props();

    const parentSignal = getAbortContext();
    const abort = createAbortContext();

    let isStreaming = $derived(generation.status === 'streaming');

    // --- Template (refine mode only) ---

    let isLoadingTemplates = $state(false);
    let isLoadingTemplateContent = $state(false);
    let templateContentError: string | null = $state(null);
    let templateListError: string | null = $state(null);
    let selectedTemplateName = $state('');
    let templateVariables: string[] = $state([]);
    let editDialogOpen = $state(false);
    let ephemeralDialogOpen = $state(false);

    // Preserves custom-template edits across picker switches. The
    // ``(custom)`` picker option has no backend identity, so without
    // this its content would be lost whenever the user picks an
    // existing template and switches back. ``templateContent`` (the
    // bindable prop) stays the *effective* content — what the host
    // snapshots for generation / history — while this holds the
    // custom draft so the ``!name`` branch below can restore it.
    let customTemplateContent = $state('');

    $effect(() => {
        // Only fetch the template list in refine mode. Generate
        // mode never reads it, so we skip the round-trip.
        if (mode !== 'refine') {
            return;
        }
        void loadTemplateList();
    });

    async function loadTemplateList() {
        isLoadingTemplates = true;
        templateListError = null;
        try {
            await refreshTemplates(abort.signal);
        } catch (e) {
            if (abort.signal.aborted) {
                return;
            }
            templateListError = friendlyErrorMessage(e, 'Failed to load templates');
        } finally {
            if (!abort.signal.aborted) {
                isLoadingTemplates = false;
            }
        }
    }

    $effect(() => {
        const name = selectedTemplateName;
        if (!name) {
            // ``(custom)`` picker option — restore the preserved
            // custom draft (empty → the card shows the placeholder).
            // Switching to an existing template doesn't touch
            // ``customTemplateContent``, so coming back to custom
            // recovers the user's edits.
            templateContent = customTemplateContent;
            return;
        }
        const controller = linkedController(parentSignal);
        isLoadingTemplateContent = true;
        templateContentError = null;
        (async () => {
            try {
                const info = await fetchTemplate(name, controller.signal);
                if (controller.signal.aborted) {
                    return;
                }
                templateContent = info.content;
            } catch (e) {
                if (controller.signal.aborted) {
                    return;
                }
                templateContentError = friendlyErrorMessage(e, 'Failed to load template');
            } finally {
                if (!controller.signal.aborted) {
                    isLoadingTemplateContent = false;
                }
            }
        })();
        return () => {
            controller.abort();
        };
    });

    // The JinjaEditor (which extracts variables via its own effect) only
    // mounts when there's content, so clearing the content elsewhere
    // (picker → (custom), or applying an empty ephemeral template)
    // would leave stale variable chips. Clear them here for any path.
    $effect(() => {
        if (!templateContent) {
            templateVariables = [];
        }
    });

    function handleEditSaved() {
        editDialogOpen = false;
        // Force the content-load effect to refetch the (now-updated)
        // template — same toggle trick caption/TemplateSection uses.
        const name = selectedTemplateName;
        selectedTemplateName = '';
        selectedTemplateName = name;
    }

    export function restoreTemplate(content: string) {
        // History entries carry content, not a template name, so restore
        // always lands on the (custom) picker with the content seeded
        // into the preserved draft. This keeps the restored content
        // consistent with the picker selection (so Edit does the right
        // thing) and lets it survive picker round-trips.
        selectedTemplateName = '';
        customTemplateContent = content;
        templateContent = content;
    }
</script>

<section class="space-y-4">
    <!-- Intent -->
    <div>
        <label class="label" for="prompts-intent">
            {mode === 'refine' ? 'Refinement intent' : 'Intent'}
        </label>
        <p class="mb-1 text-xs text-gray-500">
            {#if mode === 'refine'}
                What should change in the existing template? Be specific about which blocks to add,
                remove, or keep.
            {:else}
                What should the generated template do? Be specific about tone, length, and
                structure.
            {/if}
        </p>
        <textarea
            id="prompts-intent"
            bind:value={intent}
            onchange={() => onchange('intent', intent)}
            class="input h-28 resize-y"
            placeholder={mode === 'refine'
                ? 'e.g. Add a "trigger_warning" user block and make the system prompt more concise.'
                : 'e.g. Caption each image as a single concise sentence in the style of a museum wall label. Focus on the subject and the action, avoid adjectives.'}
            disabled={isStreaming}
        ></textarea>
    </div>

    <!-- Focus -->
    <div>
        <span class="label">Focus</span>
        <p class="mb-1 text-xs text-gray-500">
            Which blocks should the model generate? The other block is filled with a minimal
            placeholder.
        </p>
        <div class="flex gap-2">
            {#each ['both', 'system', 'user'] as const as f (f)}
                <button
                    class="cursor-pointer rounded-lg border px-3 py-1.5 text-sm transition-colors {focus ===
                    f
                        ? 'border-accent bg-accent/10 text-accent'
                        : 'border-border bg-bg text-gray-400 hover:border-gray-500 hover:text-white'}"
                    onclick={() => {
                        focus = f;
                        onchange('focus', f);
                    }}
                    disabled={isStreaming}
                >
                    {f}
                </button>
            {/each}
        </div>
    </div>

    <!-- Mode (new vs refine) -->
    <div>
        <span class="label">Mode</span>
        <p class="mb-1 text-xs text-gray-500">
            Start from scratch, or refine an existing template.
        </p>
        <div class="flex gap-2">
            {#each ['generate', 'refine'] as const as m (m)}
                <button
                    class="cursor-pointer rounded-lg border px-3 py-1.5 text-sm transition-colors {mode ===
                    m
                        ? 'border-accent bg-accent/10 text-accent'
                        : 'border-border bg-bg text-gray-400 hover:border-gray-500 hover:text-white'}"
                    onclick={() => {
                        mode = m;
                        onchange('mode', m);
                    }}
                    disabled={isStreaming}
                >
                    {m === 'refine' ? 'Refine' : 'New'}
                </button>
            {/each}
        </div>
    </div>

    <!-- Template (refine mode only) -->
    {#if mode === 'refine'}
        <div class="space-y-2">
            <div>
                <h3 class="section-heading mb-1">Template</h3>
                <p class="text-xs text-gray-500">
                    Pick a template to refine, or leave unset to generate a new one.
                </p>
            </div>
            {#if templateListError}
                <p class="text-xs text-error">{templateListError}</p>
            {/if}
            <select
                class="input cursor-pointer"
                bind:value={selectedTemplateName}
                disabled={isLoadingTemplates}
            >
                <option value="">(custom)</option>
                {#each $templates.items as t (t.name)}
                    <option value={t.name}>
                        {t.name}{#if t.source === 'builtin'}
                            (built-in){/if}
                    </option>
                {/each}
            </select>
            <!-- Card mirrors caption/TemplateSection. The editor is never
                 inline-editable — changes go through the dialog. Three
                 states: an existing template is picked (read-only editor,
                 Edit → saves back to the backend), an ephemeral template
                 is set via the dialog (read-only editor, Edit → reopens
                 the dialog without persisting), or nothing is set
                 (placeholder warning that refining now generates a new
                 template instead of refining an existing one). -->
            <Card>
                <div class="relative h-64">
                    {#if isLoadingTemplateContent}
                        <SpinnerBlock class="py-8" size="h-4 w-4" label="Loading template…" />
                    {:else if templateContentError}
                        <div class="flex h-full items-center justify-center p-4 text-sm text-error">
                            {templateContentError}
                        </div>
                    {:else if templateContent}
                        <JinjaEditor
                            class="h-full rounded-md border border-border bg-surface text-sm"
                            editable={false}
                            value={templateContent}
                            bind:variables={templateVariables}
                        />
                    {:else}
                        <div
                            class="flex h-full items-center justify-center p-4 text-center text-sm text-gray-500"
                        >
                            No template set. Refining will generate a new template instead of
                            refining an existing one.
                        </div>
                    {/if}
                </div>
                <ActionBar>
                    <ActionBarItem
                        onclick={() =>
                            selectedTemplateName
                                ? (editDialogOpen = true)
                                : (ephemeralDialogOpen = true)}
                        icon={SvgEdit}
                        variant="primary">Edit</ActionBarItem
                    >
                </ActionBar>
            </Card>
            {#if templateVariables.length > 0}
                <div class="flex flex-wrap items-center gap-1.5">
                    <span class="text-xs text-gray-500">Variables:</span>
                    {#each templateVariables as v (v)}
                        <code
                            class="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-xs text-accent"
                            >{v}</code
                        >
                    {/each}
                </div>
            {/if}
        </div>

        <!-- Edit an existing template — saves back to the backend. -->
        <EditTemplateDialog
            open={editDialogOpen}
            templateName={selectedTemplateName}
            onclose={() => (editDialogOpen = false)}
            onsaved={handleEditSaved}
        />

        <!-- Compose / edit an ephemeral template — not persisted; the
             content is applied directly to the form's templateContent. -->
        <EditTemplateDialog
            ephemeral
            open={ephemeralDialogOpen}
            templateName={null}
            initialContent={templateContent}
            onclose={() => (ephemeralDialogOpen = false)}
            onapply={(content) => {
                customTemplateContent = content;
                templateContent = content;
                ephemeralDialogOpen = false;
            }}
            onsaved={() => {}}
        />
    {/if}

    <!-- Examples -->
    <div>
        <h3 class="section-heading mb-2">Few-shot examples</h3>
        <p class="mb-2 text-xs text-gray-500">
            Optional. Adding examples dramatically improves style transfer. Each example needs an
            image, a subject, and a caption.
        </p>
        <ExamplesPanel bind:examples disabled={isStreaming} />
    </div>
</section>
