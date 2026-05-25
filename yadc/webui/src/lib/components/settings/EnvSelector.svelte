<script lang="ts">
	import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
	import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
	import { envs, refreshEnvs, fetchEnv, fetchModels, type EnvInfo } from '$lib/stores/envs';
	import { friendlyErrorMessage } from '$lib/api';

	interface Props {
		/** Increment to trigger env list reload. */
		reload?: number;
		/** Called when user clicks "Manage…". */
		onmanagerequest?: () => void;
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
		reload = 0,
		onmanagerequest,
		env: selectedEnv = $bindable('default'),
		apiUrl: envUrl = $bindable(''),
		apiToken: envToken = $bindable(''),
		apiModelName: envModelName = $bindable('')
	}: Props = $props();

	let envInfo: EnvInfo | null = $state(null);

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
			await refreshEnvs();
		} catch (e) {
			envsError = friendlyErrorMessage(e, 'Failed to load environments');
		} finally {
			isLoadingEnvs = false;
		}
	}

	// Load envs on mount and when reload changes
	$effect(() => {
		void reload;
		loadEnvs();
	});

	// Load env detail when selection changes
	$effect(() => {
		const env = selectedEnv;
		if (!env) {
			return;
		}

		let cancelled = false;
		(async () => {
			try {
				const info = await fetchEnv(env);
				if (cancelled) {
					return;
				}
				envInfo = info;
				envUrl = info.api_url || '';
				envToken = ''; // Don't pre-fill token (masked as [REDACTED] in API)
				envModelName = info.api_model_name || '';

				// Reset model list when env changes
				models = [];
				modelFetchDone = false;
				modelsError = null;

				// Auto-fetch models so the dropdown populates immediately
				if (!cancelled) {
					await loadModels();
				}
			} catch {
				if (cancelled) {
					return;
				}
				envInfo = null;
			}
		})();

		return () => {
			cancelled = true;
		};
	});

	async function loadModels() {
		if (!selectedEnv) {
			return;
		}
		isLoadingModels = true;
		modelsError = null;
		try {
			const result = await fetchModels(selectedEnv);
			models = result.models;
			modelFetchDone = true;
		} catch (e) {
			modelsError = friendlyErrorMessage(e, 'Failed to fetch models');
		} finally {
			isLoadingModels = false;
		}
	}
</script>

{#if envsError}
	<div class="alert-error">{envsError}</div>
{/if}

<section class="space-y-3">
	<div class="flex items-center justify-between">
		<h3 class="section-heading">Environment</h3>
		{#if onmanagerequest}
			<button
				class="cursor-pointer text-xs text-accent hover:text-accent-hover"
				onclick={onmanagerequest}
			>
				Manage…
			</button>
		{/if}
	</div>

	<div>
		<label class="label" for="caption-env">Environment</label>
		<select
			id="caption-env"
			class="input cursor-pointer"
			bind:value={selectedEnv}
			disabled={isLoadingEnvs}
		>
			{#each $envs.items as name (name)}
				<option value={name}>{name}</option>
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
				{#if envInfo?.api_token}
					<span class="ml-1 text-gray-500">(leave blank to use saved)</span>
				{/if}
			</label>
			<input
				id="caption-token"
				type="password"
				bind:value={envToken}
				class="input"
				placeholder={envInfo?.api_token ? '•••••••• (saved)' : 'sk-…'}
			/>
		</div>

		<!-- Model -->
		<div>
			<label class="label" for="caption-model">Model</label>
			<div class="flex gap-2">
				{#if modelFetchDone && models.length > 0}
					<select id="caption-model" class="input cursor-pointer" bind:value={envModelName}>
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
					onclick={loadModels}
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
