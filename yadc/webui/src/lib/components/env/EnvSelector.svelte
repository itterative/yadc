<script lang="ts">
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { envs, refreshEnvs, fetchModels } from '$lib/stores/env';
    import { settingsDialog } from '$lib/stores/settings';
    import { withPasswordRetry } from '$lib/stores/passwordPrompt';
    import { friendlyErrorMessage } from '$lib/api';
    import { createAbortContext, getAbortContext, linkedController } from '$lib/abort';

    interface Props {
        /** Selected environment name. */
        env?: string;
        /** API URL. */
        apiUrl?: string;
        /** API token. */
        apiToken?: string;
        /** API model name. */
        apiModelName?: string;
    }

    let {
        env: selectedEnv = $bindable('default'),
        apiUrl: envUrl = $bindable(''),
        apiToken: envToken = $bindable(''),
        apiModelName: envModelName = $bindable('')
    }: Props = $props();

    // Parent abort context — read at init time, used in effects.
    const parentSignal = getAbortContext();

    // Component-level abort scope for the env list fetch, which only runs once
    // on mount and should cancel on unmount.
    const abort = createAbortContext();

    let envInfo = $derived($envs.items.find((e) => e.name === selectedEnv) ?? null);

    let models: string[] = $state([]);
    let isLoadingModels = $state(false);
    let modelsError: string | null = $state(null);
    let modelFetchDone = $state(false);

    let isLoadingEnvs = $state(false);
    let envsError: string | null = $state(null);

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

    // Load envs on mount (SSE auto-refresh handles subsequent changes).
    // This fetch is tied to the component scope, not an effect rerun.
    $effect(() => {
        loadEnvs();
        return () => abort.abort();
    });

    // Pre-fill fields and reset models when selection changes
    $effect(() => {
        const env = selectedEnv;
        if (!env) {
            return;
        }
        const info = $envs.items.find((e) => e.name === env);
        if (info) {
            envUrl = info.api_url || '';
            envToken = ''; // Don't pre-fill token (masked as [REDACTED] in API)
            envModelName = info.api_model_name || '';
        }

        // Reset model list when env changes
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
        if (!selectedEnv) {
            return;
        }
        isLoadingModels = true;
        modelsError = null;
        try {
            // ``fetchModels`` calls the backend's ``/api/envs/<name>/models``
            // endpoint, which decrypts the env's API token. If the env is in
            // password mode and the ``yadc_password`` session cookie isn't
            // set, the backend returns 403 PASSWORD_REQUIRED — the retry
            // helper prompts the user and re-runs the call. The signal
            // aborts the prompt too, so switching envs mid-prompt doesn't
            // leave a dangling dialog.
            const result = await withPasswordRetry(() => fetchModels(selectedEnv, signal), signal);
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

<section class="space-y-3">
    <div class="flex items-center justify-between">
        <h3 class="section-heading">Environment</h3>
        <button
            class="cursor-pointer text-xs text-accent hover:text-accent-hover"
            onclick={() => settingsDialog.set({ open: true, tab: 'environments' })}
        >
            Manage…
        </button>
    </div>

    <div>
        <label class="label" for="caption-env">Environment</label>
        <select
            id="caption-env"
            class="input cursor-pointer"
            bind:value={selectedEnv}
            disabled={isLoadingEnvs}
        >
            {#each $envs.items as env (env.name)}
                <option value={env.name}>{env.name}</option>
            {/each}
            {#if $envs.items.length === 0}
                <option value="default" disabled>default</option>
            {/if}
        </select>
    </div>

    <div class="grid grid-cols-1 gap-3">
        <!-- API URL -->
        <div>
            <label class="label" for="caption-url">API URL</label>
            <input
                id="caption-url"
                type="text"
                bind:value={envUrl}
                class="input"
                placeholder="https://api.openai.com"
            />
        </div>

        <!-- API Token -->
        <div>
            <label class="label" for="caption-token">
                API Token
                {#if envInfo?.has_token}
                    <span class="ml-1 text-gray-500">(leave blank to use saved)</span>
                {/if}
            </label>
            <input
                id="caption-token"
                type="password"
                bind:value={envToken}
                class="input"
                placeholder={envInfo?.has_token ? '•••••••• (saved)' : 'sk-…'}
            />
        </div>

        {#if envInfo?.has_token}
            {#if envInfo?.token_method === 'keyring'}
                <div class="rounded-lg border border-yellow-700/50 bg-yellow-900/50 px-3 py-2">
                    <p class="text-xs text-yellow-200">
                        This environment has a keyring-encrypted API token. If your keyring is
                        locked, captioning will fail with an authentication error.
                    </p>
                </div>
            {:else if envInfo?.token_method === 'password'}
                <div class="rounded-lg border border-blue-700/50 bg-blue-900/50 px-3 py-2">
                    <p class="text-xs text-blue-200">
                        This environment has a password-encrypted API token. You'll be prompted for
                        the password the first time you use it; it's then held in a session cookie
                        for the rest of the tab. You can also sign in via Settings → Security.
                    </p>
                </div>
            {/if}
        {/if}

        <!-- Model -->
        <div>
            <label class="label" for="caption-model">Model</label>
            <div class="flex gap-2">
                {#if modelFetchDone && models.length > 0}
                    <select
                        id="caption-model"
                        class="input cursor-pointer"
                        bind:value={envModelName}
                    >
                        {#each models as m (m)}
                            <option value={m}>{m}</option>
                        {/each}
                        {#if !models.includes(envModelName) && envModelName}
                            <option value={envModelName}>{envModelName}</option>
                        {/if}
                    </select>
                {:else}
                    <input
                        id="caption-model"
                        type="text"
                        bind:value={envModelName}
                        class="input"
                        placeholder="gpt-4o-mini"
                    />
                {/if}
                <button
                    class="cursor-pointer rounded-lg border border-border bg-bg px-3 py-2 text-gray-400 transition-colors hover:border-gray-500 hover:text-white disabled:opacity-50"
                    onclick={() => loadModels()}
                    disabled={isLoadingModels || !selectedEnv}
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
</section>
