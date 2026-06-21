<script lang="ts">
    import { templates, refreshTemplates, fetchTemplate } from '$lib/stores/templates';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, getAbortContext, linkedController } from '$lib/abort';
    import { generation } from '$lib/stores/prompts';
    import type { ExamplePair, PromptGenFocus, PromptGenMode } from '$lib/stores/prompts';
    import JinjaEditor from '$lib/components/ui/JinjaEditor.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
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
        <div>
            <h3 class="section-heading mb-2">Template</h3>
            <p class="mb-2 text-xs text-gray-500">
                Pick the template to refine, or leave blank and paste your own content below.
            </p>
            {#if templateListError}
                <p class="mb-2 text-xs text-error">{templateListError}</p>
            {/if}
            <div class="space-y-2">
                <select
                    class="input cursor-pointer"
                    bind:value={selectedTemplateName}
                    disabled={isLoadingTemplates}
                >
                    <option value="">(no template — paste your own)</option>
                    {#each $templates.items as t (t.name)}
                        <option value={t.name}>
                            {t.name}{#if t.source === 'builtin'}
                                (built-in){/if}
                        </option>
                    {/each}
                </select>
                <div class="relative h-64">
                    {#if isLoadingTemplateContent}
                        <SpinnerBlock class="py-8" size="h-4 w-4" label="Loading template…" />
                    {:else if templateContentError}
                        <div class="flex h-full items-center justify-center p-4 text-sm text-error">
                            {templateContentError}
                        </div>
                    {:else}
                        <JinjaEditor
                            class="h-full rounded-md border border-border bg-surface text-sm"
                            bind:value={templateContent}
                            bind:variables={templateVariables}
                        />
                    {/if}
                </div>
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
        </div>
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
