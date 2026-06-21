<script lang="ts">
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { envs, refreshEnvs, fetchModels } from '$lib/stores/env';
    import { withPasswordRetry } from '$lib/stores/passwordPrompt';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, getAbortContext, linkedController } from '$lib/abort';
    import { generation } from '$lib/stores/prompts';
    import type { PromptImageQuality } from '$lib/stores/prompts';

    interface Props {
        env: string;
        apiUrl: string;
        apiToken: string;
        apiModelName: string;
        /** Output token cap for the generation call. */
        maxTokens: number;
        /** Few-shot example image fidelity. */
        imageQuality: PromptImageQuality;
        onchange: (field: string, value: unknown) => void;
    }

    let {
        env = $bindable('default'),
        apiUrl = $bindable(''),
        apiToken = $bindable(''),
        apiModelName = $bindable(''),
        maxTokens = $bindable(16384),
        imageQuality = $bindable('auto'),
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

    // Pre-fill apiUrl / apiToken / apiModelName from the selected env
    // and fetch its model list. Runs on first mount (applies the env
    // defaults) and whenever the env selection changes.
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
                    disabled={isLoadingEnvs || isStreaming}
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
                        disabled={isStreaming}
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
                        disabled={isStreaming}
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
                                disabled={isStreaming}
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
                                disabled={isStreaming}
                            />
                        {/if}
                        <button
                            class="cursor-pointer rounded-lg border border-border bg-bg px-3 py-2 text-gray-400 transition-colors hover:border-gray-500 hover:text-white disabled:opacity-50"
                            onclick={() => loadModels()}
                            disabled={isLoadingModels || !env || isStreaming}
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

    <!-- Generation limits -->
    <div>
        <h3 class="section-heading mb-2">Generation limits</h3>
        <div class="space-y-4">
            <div>
                <label class="label" for="prompts-max-tokens">Max tokens</label>
                <p class="mb-1 text-xs text-gray-500">
                    Output token cap for the generation call. The default of 4096 truncates longer
                    templates — raise this if output gets cut off (lower it for models with small
                    output windows).
                </p>
                <input
                    id="prompts-max-tokens"
                    type="number"
                    min="100"
                    max="65536"
                    step="512"
                    bind:value={maxTokens}
                    onchange={() => onchange('maxTokens', maxTokens)}
                    class="input w-40"
                    disabled={isStreaming}
                />
            </div>

            <div>
                <span class="label">Image quality</span>
                <p class="mb-1 text-xs text-gray-500">
                    Few-shot example image fidelity. Maps to OpenAI's <code>image_url.detail</code>
                    and Gemini's <code>mediaResolution</code>. Higher costs more tokens.
                </p>
                <div class="flex gap-2">
                    {#each ['auto', 'low', 'high'] as const as q (q)}
                        <button
                            class="cursor-pointer rounded-lg border px-3 py-1.5 text-sm transition-colors {imageQuality ===
                            q
                                ? 'border-accent bg-accent/10 text-accent'
                                : 'border-border bg-bg text-gray-400 hover:border-gray-500 hover:text-white'}"
                            onclick={() => {
                                imageQuality = q;
                                onchange('imageQuality', q);
                            }}
                            disabled={isStreaming}
                        >
                            {q}
                        </button>
                    {/each}
                </div>
            </div>
        </div>
    </div>
</section>
