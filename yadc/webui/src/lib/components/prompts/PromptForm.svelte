<script lang="ts">
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { envs, refreshEnvs, fetchModels } from '$lib/stores/env';
    import { templates, refreshTemplates, fetchTemplate } from '$lib/stores/templates';
    import { withPasswordRetry } from '$lib/stores/passwordPrompt';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, getAbortContext, linkedController } from '$lib/abort';
    import { generation } from '$lib/stores/prompts';
    import type { ExamplePair, PromptGenFocus, PromptGenMode } from '$lib/stores/prompts';
    import JinjaEditor from '$lib/components/ui/JinjaEditor.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import ExamplesPanel from './ExamplesPanel.svelte';

    interface Props {
        /** Generate vs refine mode. Read-only — the host owns the
         *  state and passes it down. */
        mode: PromptGenMode;
        env: string;
        apiUrl: string;
        apiToken: string;
        apiModelName: string;
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
        mode,
        env = $bindable('default'),
        apiUrl = $bindable(''),
        apiToken = $bindable(''),
        apiModelName = $bindable(''),
        intent = $bindable(''),
        focus = $bindable('both'),
        examples = $bindable([]),
        templateContent = $bindable(''),
        onchange
    }: Props = $props();

    const parentSignal = getAbortContext();
    const abort = createAbortContext();

    let envInfo = $derived($envs.items.find((e) => e.name === env) ?? null);

    let models: string[] = $state([]);
    let isLoadingModels = $state(false);
    let modelsError: string | null = $state(null);
    let modelFetchDone = $state(false);

    let isLoadingEnvs = $state(false);
    let envsError: string | null = $state(null);

    let isStreaming = $derived(generation.status === 'streaming');

    async function loadEnvs() {
        isLoadingEnvs = true;
        envsError = null;
        try {
            await refreshEnvs(abort.signal);
        } catch (e) {
            if (abort.signal.aborted) {
                return;
            }
            envsError = friendlyErrorMessage(e, 'Failed to load environments');
        } finally {
            if (!abort.signal.aborted) {
                isLoadingEnvs = false;
            }
        }
    }

    $effect(() => {
        loadEnvs();
        return () => abort.abort();
    });

    // Pre-fill fields and reset models when env changes. Re-runs on
    // every mount of this component (so also when the host re-mounts
    // the form by switching to the other tab). The field reset is the
    // same values the user already sees in the other tab (both forms
    // share state via bind), so the reset is effectively a no-op
    // except on the very first mount of the very first time the page
    // is opened — which is exactly when the env defaults should be
    // applied. (The user accepted the double-mount trade-off; see
    // the Phase 6b history entry.)
    $effect(() => {
        const current = env;
        if (!current) {
            return;
        }
        const info = $envs.items.find((e) => e.name === current);
        if (info) {
            apiUrl = info.api_url || '';
            apiToken = '';
            apiModelName = info.api_model_name || '';
        }
        models = [];
        modelFetchDone = false;
        modelsError = null;

        const controller = linkedController(parentSignal);
        (async () => {
            if (controller.signal.aborted) {
                return;
            }
            await loadModels(controller.signal);
        })();

        return () => {
            controller.abort();
        };
    });

    async function loadModels(signal?: AbortSignal) {
        if (!env) {
            return;
        }
        isLoadingModels = true;
        modelsError = null;
        try {
            const result = await withPasswordRetry(() => fetchModels(env, signal), signal);
            if (signal?.aborted) {
                return;
            }
            models = result.models;
            modelFetchDone = true;
        } catch (e) {
            if (signal?.aborted) {
                return;
            }
            modelsError = friendlyErrorMessage(e, 'Failed to fetch models');
        } finally {
            if (!signal?.aborted) {
                isLoadingModels = false;
            }
        }
    }

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

{#if envsError}
    <div class="alert-error">{envsError}</div>
{/if}

<section class="space-y-4">
    <!-- Env + model -->
    <div>
        <h3 class="section-heading mb-2">Environment</h3>
        <div class="space-y-3">
            <div>
                <label class="label" for="prompts-env">Environment</label>
                <select
                    id="prompts-env"
                    class="input cursor-pointer"
                    bind:value={env}
                    onchange={() => onchange('env', env)}
                    disabled={isLoadingEnvs}
                >
                    {#each $envs.items as e (e.name)}
                        <option value={e.name}>{e.name}</option>
                    {/each}
                    {#if $envs.items.length === 0}
                        <option value="default" disabled>default</option>
                    {/if}
                </select>
            </div>

            <div class="grid grid-cols-1 gap-3">
                <div>
                    <label class="label" for="prompts-url">API URL</label>
                    <input
                        id="prompts-url"
                        type="text"
                        bind:value={apiUrl}
                        onchange={() => onchange('apiUrl', apiUrl)}
                        class="input"
                        placeholder="https://api.openai.com"
                    />
                </div>
                <div>
                    <label class="label" for="prompts-token">
                        API Token
                        {#if envInfo?.has_token}
                            <span class="ml-1 text-gray-500">(leave blank to use saved)</span>
                        {/if}
                    </label>
                    <input
                        id="prompts-token"
                        type="password"
                        bind:value={apiToken}
                        onchange={() => onchange('apiToken', apiToken)}
                        class="input"
                        placeholder={envInfo?.has_token ? '•••••••• (saved)' : 'sk-…'}
                    />
                </div>
                <div>
                    <label class="label" for="prompts-model">Model</label>
                    <div class="flex gap-2">
                        {#if modelFetchDone && models.length > 0}
                            <select
                                id="prompts-model"
                                class="input cursor-pointer"
                                bind:value={apiModelName}
                                onchange={() => onchange('apiModelName', apiModelName)}
                            >
                                {#each models as m (m)}
                                    <option value={m}>{m}</option>
                                {/each}
                                {#if !models.includes(apiModelName) && apiModelName}
                                    <option value={apiModelName}>{apiModelName}</option>
                                {/if}
                            </select>
                        {:else}
                            <input
                                id="prompts-model"
                                type="text"
                                bind:value={apiModelName}
                                onchange={() => onchange('apiModelName', apiModelName)}
                                class="input"
                                placeholder="gpt-4o-mini"
                            />
                        {/if}
                        <button
                            class="cursor-pointer rounded-lg border border-border bg-bg px-3 py-2 text-gray-400 transition-colors hover:border-gray-500 hover:text-white disabled:opacity-50"
                            onclick={() => loadModels()}
                            disabled={isLoadingModels || !env}
                            title="Fetch available models"
                        >
                            {#if isLoadingModels}
                                <SvgSpinner class="h-4 w-4 animate-spin" />
                            {:else}
                                <SvgRefresh class="h-4 w-4" />
                            {/if}
                        </button>
                    </div>
                    {#if modelsError}
                        <p class="mt-1 text-xs text-error">{modelsError}</p>
                    {/if}
                </div>
            </div>
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
