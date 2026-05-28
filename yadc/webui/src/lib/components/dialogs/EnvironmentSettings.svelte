<script lang="ts">
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SvgVisibility from '$lib/icons/SvgVisibility.svelte';
    import SvgVisibilityOff from '$lib/icons/SvgVisibilityOff.svelte';
    import ConfirmDelete from '$lib/components/ui/ConfirmDelete.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import {
        deleteEnv,
        refreshEnvs,
        revealEnvValue,
        saveEnv,
        envs,
        type EnvInfo
    } from '$lib/stores/envs';
    import { friendlyErrorMessage, PasswordRequiredError } from '$lib/api';
    import {
        withPasswordRetry,
        PasswordPromptCancelled,
        cancelPassword
    } from '$lib/stores/passwordPrompt';

    let envLoading = $state(false);
    let envError: string | null = $state(null);
    let editingEnv: EnvInfo | null = $state(null);
    let isNewEnv = $state(false);
    let editName = $state('');
    let editUrl = $state('');
    let editToken = $state('');
    let editModelName = $state('');
    let isSavingEnv = $state(false);
    let saveEnvError: string | null = $state(null);
    let confirmDelete: string | null = $state(null);

    let showToken = $state(false);
    let revealedToken = $state('');

    $effect(() => {
        loadEnvs();
    });

    async function loadEnvs() {
        envLoading = true;
        envError = null;
        try {
            await refreshEnvs();
        } catch (e) {
            envError = friendlyErrorMessage(e, 'Failed to load environments');
        } finally {
            envLoading = false;
        }
    }

    function startCreateEnv() {
        isNewEnv = true;
        editingEnv = null;
        editName = '';
        editUrl = '';
        editToken = '';
        editModelName = '';
        saveEnvError = null;
        showToken = false;
        revealedToken = '';
    }

    function startEditEnv(env: EnvInfo) {
        saveEnvError = null;
        showToken = false;
        revealedToken = '';
        isNewEnv = false;
        editingEnv = env;
        editName = env.name;
        editUrl = env.api_url || '';
        editToken = '';
        editModelName = env.api_model_name || '';
    }

    function cancelEditEnv() {
        editingEnv = null;
        isNewEnv = false;
        saveEnvError = null;
        showToken = false;
        revealedToken = '';
        cancelPassword();
    }

    async function handleSaveEnv() {
        const name = editName.trim();
        if (!name) {
            saveEnvError = 'Name is required';
            return;
        }

        isSavingEnv = true;
        saveEnvError = null;
        try {
            const data: Record<string, string> = { api_url: editUrl.trim() };
            if (editToken) {
                data.api_token = editToken;
            }
            if (editModelName.trim()) {
                data.api_model_name = editModelName.trim();
            }

            await saveEnv(name, data);
            editingEnv = null;
            isNewEnv = false;
            await loadEnvs();
        } catch (e) {
            saveEnvError = friendlyErrorMessage(e, 'Failed to save environment');
        } finally {
            isSavingEnv = false;
        }
    }

    async function handleDeleteEnv(name: string) {
        try {
            await deleteEnv(name);
            confirmDelete = null;
            await loadEnvs();
        } catch (e) {
            envError = friendlyErrorMessage(e, 'Failed to delete environment');
        }
    }

    async function toggleTokenReveal() {
        if (showToken) {
            // If user didn't change the revealed token, clear it so placeholder shows
            if (editToken === revealedToken) {
                editToken = '';
            }
            revealedToken = '';
            showToken = false;
            return;
        }

        // If the user already has a token value (typed or modified), just toggle visibility
        if (editToken) {
            showToken = true;
            return;
        }

        const envName = editingEnv?.name;
        if (!envName) {
            return;
        }

        try {
            const data = await withPasswordRetry(() => revealEnvValue(envName, 'api_token'));
            editToken = data.value;
            revealedToken = data.value;
            showToken = true;
        } catch (e) {
            if (e instanceof PasswordPromptCancelled) {
                return;
            }
            if (e instanceof PasswordRequiredError) {
                saveEnvError = 'Incorrect password. Please try again.';
                return;
            }
            saveEnvError = friendlyErrorMessage(e, 'Failed to reveal token');
        }
    }
</script>

<div class="space-y-4 p-5">
    <p class="text-sm text-gray-500">
        Environments store API connection settings (URL, token, and default model) so you can
        quickly switch between different providers or local servers when captioning.
    </p>

    {#if envError}
        <div class="alert-error">{envError}</div>
    {/if}

    {#if !editingEnv && !isNewEnv}
        <div class="space-y-2">
            {#if envLoading}
                <SpinnerBlock class="py-8" />
            {:else if $envs.items.length === 0}
                <p class="py-4 text-center text-sm text-gray-500">No environments yet.</p>
            {:else}
                {#each $envs.items as env (env.name)}
                    <div
                        class="group flex items-center gap-3 rounded-lg border border-border bg-bg p-3"
                    >
                        <div class="min-w-0 flex-1">
                            <div class="flex items-center gap-2">
                                <span class="text-sm font-medium text-white">{env.name}</span>
                                {#if env.api_url}
                                    <span
                                        class="rounded border border-gray-700/40 bg-gray-800/40 px-1.5 py-px text-[0.625rem] font-medium text-gray-400"
                                    >
                                        URL
                                    </span>
                                {/if}
                                {#if env.has_token}
                                    <span
                                        class="rounded border border-accent/25 bg-accent/15 px-1.5 py-px text-[0.625rem] font-medium text-accent"
                                    >
                                        Token
                                    </span>
                                {/if}
                                {#if env.api_model_name}
                                    <span
                                        class="rounded border border-gray-700/40 bg-gray-800/40 px-1.5 py-px text-[0.625rem] font-medium text-gray-400"
                                    >
                                        Model
                                    </span>
                                {/if}
                            </div>
                        </div>
                        <div
                            class="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 max-lg:opacity-100"
                        >
                            <button
                                class="cursor-pointer rounded p-1.5 text-gray-400 transition-colors hover:text-accent"
                                title="Edit"
                                onclick={() => startEditEnv(env)}
                            >
                                <SvgEdit class="h-4 w-4" />
                            </button>
                            {#if env.name !== 'default'}
                                <button
                                    class="cursor-pointer rounded p-1.5 text-gray-400 transition-colors hover:text-error"
                                    title="Delete"
                                    onclick={() => (confirmDelete = env.name)}
                                >
                                    <SvgDelete class="h-4 w-4" />
                                </button>
                            {/if}
                        </div>
                    </div>
                {/each}
            {/if}
        </div>

        <button
            class="btn-secondary w-full justify-center py-2.5 text-gray-200"
            onclick={startCreateEnv}
        >
            <SvgPlus class="h-4 w-4" />
            New Environment
        </button>

        <ConfirmDelete
            open={confirmDelete !== null}
            oncancel={() => (confirmDelete = null)}
            onconfirm={() => handleDeleteEnv(confirmDelete!)}
        >
            Delete environment <strong>{confirmDelete}</strong>?
        </ConfirmDelete>
    {:else}
        <div class="space-y-4">
            {#if saveEnvError}
                <div class="alert-error">{saveEnvError}</div>
            {/if}

            <div>
                <label class="label" for="env-name">Name</label>
                <input
                    id="env-name"
                    type="text"
                    bind:value={editName}
                    disabled={!isNewEnv}
                    class="input disabled:opacity-50"
                    placeholder="my-environment"
                />
            </div>

            <div>
                <label class="label" for="env-url">API URL</label>
                <input
                    id="env-url"
                    type="text"
                    bind:value={editUrl}
                    class="input"
                    placeholder="https://api.openai.com"
                />
            </div>

            <div>
                <label class="label" for="env-token">
                    API Token
                    {#if editingEnv?.api_token}
                        <span class="ml-1 text-gray-500">(leave blank to keep current)</span>
                    {/if}
                </label>
                <div class="flex gap-2">
                    <input
                        id="env-token"
                        type={showToken ? 'text' : 'password'}
                        bind:value={editToken}
                        class="input flex-1"
                        placeholder={editingEnv?.api_token ? '••••••••' : 'sk-...'}
                    />
                    {#if editingEnv?.has_token}
                        <button
                            type="button"
                            class="btn-secondary px-3"
                            title={showToken ? 'Hide token' : 'Reveal token'}
                            onclick={toggleTokenReveal}
                            disabled={isSavingEnv}
                        >
                            {#if showToken}
                                <SvgVisibilityOff class="h-4 w-4" />
                            {:else}
                                <SvgVisibility class="h-4 w-4" />
                            {/if}
                        </button>
                    {/if}
                </div>
            </div>

            <div>
                <label class="label" for="env-model">Default Model</label>
                <input
                    id="env-model"
                    type="text"
                    bind:value={editModelName}
                    class="input"
                    placeholder="gpt-4o-mini"
                />
            </div>

            <div class="btn-bar">
                <button class="btn-secondary" onclick={cancelEditEnv} disabled={isSavingEnv}>
                    Cancel
                </button>
                <button class="btn-primary" onclick={handleSaveEnv} disabled={isSavingEnv}>
                    {isSavingEnv ? 'Saving...' : isNewEnv ? 'Create' : 'Save'}
                </button>
            </div>
        </div>
    {/if}
</div>
