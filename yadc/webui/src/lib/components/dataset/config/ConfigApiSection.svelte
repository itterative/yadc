<script lang="ts">
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { envs, refreshEnvs, fetchModels } from '$lib/stores/env';
    import { settingsDialog } from '$lib/stores/settings';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, getAbortContext, linkedController } from '$lib/abort';
    import { configState } from './state.svelte';

    let isLoadingEnvs = $state(false);
    let envsError: string | null = $state(null);

    // Parent abort context — read at init time, used in effects.
    const parentSignal = getAbortContext();

    // Component-level abort scope for the env list fetch, which has no
    // per-effect reruns and should cancel on unmount.
    const abort = createAbortContext();

    let models: string[] = $state([]);
    let isLoadingModels = $state(false);
    let modelsError: string | null = $state(null);
    let modelFetchDone = $state(false);

    let envInfo = $derived($envs.items.find((e) => e.name === configState.envName) ?? null);

    // Load envs on mount (SSE auto-refresh handles subsequent changes).
    // This fetch is tied to the component scope, not an effect rerun.
    $effect(() => {
        void (async () => {
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
        })();
        return () => abort.abort();
    });

    // Fetch models when the selected env changes. Only update placeholders
    // and the model dropdown — do not pre-fill the bound URL/model fields,
    // so user-typed TOML values stay intact until saved.
    $effect(() => {
        const env = configState.envName;
        models = [];
        modelFetchDone = false;
        modelsError = null;
        if (!env) {
            return;
        }
        const controller = linkedController(parentSignal);
        void (async () => {
            if (controller.signal.aborted) {
                return;
            }
            isLoadingModels = true;
            try {
                const result = await fetchModels(env, controller.signal);
                if (controller.signal.aborted) {
                    return;
                }
                models = result.models;
                modelFetchDone = true;
            } catch (e) {
                if (controller.signal.aborted) {
                    return;
                }
                modelsError = friendlyErrorMessage(e, 'Failed to fetch models');
            } finally {
                if (!controller.signal.aborted) {
                    isLoadingModels = false;
                }
            }
        })();
        return () => {
            controller.abort();
        };
    });

    async function refreshModels() {
        const env = configState.envName;
        if (!env) {
            return;
        }
        const controller = linkedController(parentSignal);
        isLoadingModels = true;
        modelsError = null;
        try {
            const result = await fetchModels(env, controller.signal);
            if (controller.signal.aborted) {
                return;
            }
            models = result.models;
            modelFetchDone = true;
        } catch (e) {
            if (controller.signal.aborted) {
                return;
            }
            modelsError = friendlyErrorMessage(e, 'Failed to fetch models');
        } finally {
            if (!controller.signal.aborted) {
                isLoadingModels = false;
            }
        }
    }
</script>

{#if envsError}
    <div class="alert-error">{envsError}</div>
{/if}

<section class="space-y-3">
    <div class="flex items-center justify-between">
        <h3 class="section-heading">API</h3>
        <button
            class="cursor-pointer text-xs text-accent hover:text-accent-hover"
            onclick={() => settingsDialog.set({ open: true, tab: 'environments' })}
        >
            Manage…
        </button>
    </div>

    <!-- Environment -->
    <div>
        <label class="label" for="config-env">Environment</label>
        <select
            id="config-env"
            class="input cursor-pointer"
            bind:value={configState.envName}
            disabled={isLoadingEnvs}
        >
            <option value="">(none)</option>
            {#each $envs.items as env (env.name)}
                <option value={env.name}>{env.name}</option>
            {/each}
        </select>
        <p class="help-text">Set this to configure API settings outside the TOML file.</p>
    </div>

    <!-- API URL -->
    <div>
        <label class="label" for="config-api-url">API URL</label>
        <input
            id="config-api-url"
            type="text"
            bind:value={configState.apiUrl}
            class="input"
            placeholder={envInfo?.api_url || 'http://localhost:11434'}
        />
        <p class="help-text">Hostname and port only — no additional path segments.</p>
        <p class="help-text">Leave empty to use the env's default URL.</p>
    </div>

    <!-- Model -->
    <div>
        <label class="label" for="config-model">Model Name</label>
        <div class="flex gap-2">
            {#if modelFetchDone && models.length > 0}
                <select
                    id="config-model"
                    class="input cursor-pointer"
                    bind:value={configState.apiModelName}
                >
                    <option value="">(env default)</option>
                    <optgroup label="Available">
                        {#each models as m (m)}
                            <option value={m}>{m}</option>
                        {/each}
                        {#if !models.includes(configState.apiModelName) && configState.apiModelName}
                            <option value={configState.apiModelName}
                                >{configState.apiModelName}</option
                            >
                        {/if}
                    </optgroup>
                </select>
            {:else}
                <input
                    id="config-model"
                    type="text"
                    bind:value={configState.apiModelName}
                    class="input"
                    placeholder={envInfo?.api_model_name || 'gemma3'}
                />
            {/if}
            <button
                class="cursor-pointer rounded-lg border border-border bg-bg px-3 py-2 text-gray-400 transition-colors hover:border-gray-500 hover:text-white disabled:opacity-50"
                onclick={refreshModels}
                disabled={isLoadingModels || !configState.envName}
                title={configState.envName
                    ? 'Fetch available models from this env'
                    : 'Select an environment to fetch models'}
            >
                {#if isLoadingModels}
                    <SvgSpinner class="h-4 w-4 animate-spin" />
                {:else}
                    <SvgRefresh class="h-4 w-4" />
                {/if}
            </button>
        </div>
        {#if !modelFetchDone || models.length === 0}
            <p class="help-text">Leave empty to use the env's default model.</p>
        {/if}
        {#if modelsError}
            <p class="mt-1 text-xs text-error">{modelsError}</p>
        {/if}
    </div>
</section>
