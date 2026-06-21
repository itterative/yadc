<script lang="ts">
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { envs, refreshEnvs, fetchModels } from '$lib/stores/env';
    import { withPasswordRetry } from '$lib/stores/passwordPrompt';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, getAbortContext, linkedController } from '$lib/abort';
    import { generation } from '$lib/stores/prompts';
    import type { ExamplePair, PromptGenFocus } from '$lib/stores/prompts';
    import ExamplesPanel from './ExamplesPanel.svelte';

    interface Props {
        env: string;
        apiUrl: string;
        apiToken: string;
        apiModelName: string;
        intent: string;
        focus: PromptGenFocus;
        examples: ExamplePair[];
        onchange: (field: string, value: unknown) => void;
    }

    let {
        env = $bindable('default'),
        apiUrl = $bindable(''),
        apiToken = $bindable(''),
        apiModelName = $bindable(''),
        intent = $bindable(''),
        focus = $bindable('both'),
        examples = $bindable([]),
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

    // Pre-fill fields and reset models when env changes
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

    <!-- Intent -->
    <div>
        <label class="label" for="prompts-intent">Intent</label>
        <p class="mb-1 text-xs text-gray-500">
            What should the generated template do? Be specific about tone, length, and structure.
        </p>
        <textarea
            id="prompts-intent"
            bind:value={intent}
            onchange={() => onchange('intent', intent)}
            class="input h-28 resize-y"
            placeholder="e.g. Caption each image as a single concise sentence in the style of a museum wall label. Focus on the subject and the action, avoid adjectives."
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
