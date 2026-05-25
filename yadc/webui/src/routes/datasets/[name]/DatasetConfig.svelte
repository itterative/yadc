<script lang="ts">
	import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
	import Checkbox from '$lib/components/ui/Checkbox.svelte';
	import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
	import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
	import { fetchConfig, patchConfig, previewConfig } from '$lib/stores/configs';
	import { templates, refreshTemplates } from '$lib/stores/templates';
	import { toast } from '$lib/stores/toasts';
	import { friendlyErrorMessage } from '$lib/api';

	// --- Props ---

	interface Props {
		/** Which dataset this is for. */
		datasetName: string;
		/** Called after the config is saved and rescanned. */
		onsaved?: () => void;
	}

	let { datasetName, onsaved }: Props = $props();

	// --- State ---

	let isLoading = $state(false);
	let isSaving = $state(false);
	let error: string | null = $state(null);
	let saveError: string | null = $state(null);

	// (template list comes from the shared store)

	// --- Structured fields (parsed from config) ---

	// API
	let apiUrl = $state('');
	let apiModelName = $state('');

	// Prompt
	let promptName = $state('');

	// Options (nullable fields: maxTokens, imageQuality, rounds — null means "not in TOML")
	let maxTokens: number | null = $state(null);
	let imageQuality: 'auto' | 'high' | 'low' | null = $state(null);
	let rounds: number | null = $state(null);
	let overwrite = $state(false);
	let storeConversation = $state(false);

	// Reasoning
	let reasoningEnabled = $state(false);
	let reasoningEffort: 'low' | 'medium' | 'high' = $state('low');
	let reasoningExcludeOutput = $state(true);

	// Environment
	let envName = $state('');

	// --- Snapshot of last-loaded values for dirty tracking ---

	let loadedApiUrl = $state('');
	let loadedApiModelName = $state('');
	let loadedPromptName = $state('');
	let loadedMaxTokens: number | null = $state(null);
	let loadedImageQuality: 'auto' | 'high' | 'low' | null = $state(null);
	let loadedRounds: number | null = $state(null);
	let loadedOverwrite = $state(false);
	let loadedStoreConversation = $state(false);
	let loadedReasoningEnabled = $state(false);
	let loadedReasoningEffort: 'low' | 'medium' | 'high' = $state('low');
	let loadedReasoningExcludeOutput = $state(true);
	let loadedEnvName = $state('');

	let dirty = $derived(
		apiUrl !== loadedApiUrl ||
			apiModelName !== loadedApiModelName ||
			promptName !== loadedPromptName ||
			maxTokens !== loadedMaxTokens ||
			imageQuality !== loadedImageQuality ||
			rounds !== loadedRounds ||
			overwrite !== loadedOverwrite ||
			storeConversation !== loadedStoreConversation ||
			reasoningEnabled !== loadedReasoningEnabled ||
			reasoningEffort !== loadedReasoningEffort ||
			reasoningExcludeOutput !== loadedReasoningExcludeOutput ||
			envName !== loadedEnvName
	);

	// --- Live preview ---

	let previewContent = $state('');
	let previewTimer: ReturnType<typeof setTimeout> | null = null;
	const DEBOUNCE_MS = 400;

	/** Build the patch dict from current field values. Nullable fields are omitted when null. */
	function buildPatch(): Record<string, unknown> {
		const settings: Record<string, unknown> = {
			store_conversation: storeConversation
		};
		if (maxTokens !== null) {
			settings.max_tokens = maxTokens;
		}
		if (imageQuality !== null) {
			settings.image_quality = imageQuality;
		}

		const patch: Record<string, unknown> = {
			api: {
				url: apiUrl,
				model_name: apiModelName
			},
			prompt: {
				name: promptName
			},
			settings,
			overwrite_captions: overwrite,
			reasoning: {
				enable: reasoningEnabled,
				thinking_effort: reasoningEffort,
				exclude_from_output: reasoningExcludeOutput
			},
			env: envName
		};
		if (rounds !== null) {
			patch.rounds = rounds;
		}
		return patch;
	}

	/** Debounced: request a dry-run preview from the backend. */
	function schedulePreview() {
		if (previewTimer !== null) {
			clearTimeout(previewTimer);
		}
		previewTimer = setTimeout(() => {
			previewTimer = null;
			previewConfig(datasetName, buildPatch())
				.then((result) => {
					previewContent = result.content;
				})
				.catch(() => {
					/* preview failure is non-critical */
				});
		}, DEBOUNCE_MS);
	}

	// --- Load on mount / dataset change ---

	$effect(() => {
		const name = datasetName;
		if (!name) {
			return;
		}
		loadConfig();
	});

	async function loadConfig() {
		isLoading = true;
		error = null;

		try {
			const templatesReady = $templates.loaded ? Promise.resolve() : refreshTemplates();

			const [config] = await Promise.all([fetchConfig(datasetName), templatesReady]);
			previewContent = config.content;
			const p = config.parsed as Record<string, unknown>;
			const api = (p.api as Record<string, unknown>) ?? {};
			const settings = (p.settings as Record<string, unknown>) ?? {};
			const reasoning = (p.reasoning as Record<string, unknown>) ?? {};
			const prompt = (p.prompt as Record<string, unknown>) ?? {};

			apiUrl = loadedApiUrl = (api.url as string) ?? '';
			apiModelName = loadedApiModelName = (api.model_name as string) ?? '';
			promptName = loadedPromptName = (prompt.name as string) ?? '';
			maxTokens = loadedMaxTokens = ('max_tokens' in settings ? settings.max_tokens : null) as
				| number
				| null;
			imageQuality = loadedImageQuality = (
				'image_quality' in settings ? settings.image_quality : null
			) as 'auto' | 'high' | 'low' | null;
			rounds = loadedRounds = ('rounds' in p ? p.rounds : null) as number | null;
			overwrite = loadedOverwrite = (p.overwrite_captions as boolean) ?? false;
			storeConversation = loadedStoreConversation =
				(settings.store_conversation as boolean) ?? false;
			reasoningEnabled = loadedReasoningEnabled = (reasoning.enable as boolean) ?? false;
			reasoningEffort = loadedReasoningEffort =
				(reasoning.thinking_effort as 'low' | 'medium' | 'high') ?? 'low';
			reasoningExcludeOutput = loadedReasoningExcludeOutput =
				(reasoning.exclude_from_output as boolean) ?? true;
			envName = loadedEnvName = (p.env as string) ?? '';
		} catch (e) {
			error = friendlyErrorMessage(e, 'Failed to load config');
		} finally {
			isLoading = false;
		}
	}

	// --- Trigger live preview on any field change ---

	$effect(() => {
		// Touch all reactive values so Svelte tracks them
		void apiUrl;
		void apiModelName;
		void promptName;
		void maxTokens;
		void imageQuality;
		void rounds;
		void overwrite;
		void storeConversation;
		void reasoningEnabled;
		void reasoningEffort;
		void reasoningExcludeOutput;
		void envName;
		void datasetName;

		if (!isLoading && !error) {
			schedulePreview();
		}
	});

	// --- Save structured fields via PATCH ---

	async function handleSave() {
		isSaving = true;
		saveError = null;

		try {
			const result = await patchConfig(datasetName, buildPatch());
			previewContent = result.content;
			// Update loaded snapshot so dirty resets
			loadedApiUrl = apiUrl;
			loadedApiModelName = apiModelName;
			loadedPromptName = promptName;
			loadedMaxTokens = maxTokens as number | null;
			loadedImageQuality = imageQuality as 'auto' | 'high' | 'low' | null;
			loadedRounds = rounds as number | null;
			loadedOverwrite = overwrite;
			loadedStoreConversation = storeConversation;
			loadedReasoningEnabled = reasoningEnabled;
			loadedReasoningEffort = reasoningEffort;
			loadedReasoningExcludeOutput = reasoningExcludeOutput;
			loadedEnvName = envName;
			toast.success('Config saved');
			onsaved?.();
		} catch (e) {
			saveError = friendlyErrorMessage(e, 'Failed to save config');
		} finally {
			isSaving = false;
		}
	}
</script>

<div class="flex h-full flex-col">
	<!-- Scrollable content -->
	<div class="flex-1 space-y-5 overflow-y-auto p-4">
		{#if isLoading}
			<SpinnerBlock class="py-8" size="h-5 w-5" label="Loading config…" />
		{:else if error}
			<div class="alert-error">{error}</div>
		{:else}
			{#if saveError}
				<div class="alert-error">{saveError}</div>
			{/if}

			<!-- ═══ Section: API ═══ -->
			<section class="space-y-3">
				<h3 class="section-heading">API</h3>

				<div class="grid grid-cols-2 gap-3">
					<div class="col-span-2">
						<label class="label" for="config-api-url">API URL</label>
						<input
							id="config-api-url"
							type="text"
							bind:value={apiUrl}
							class="input"
							placeholder="http://localhost:11434"
						/>
						<p class="help-text">Hostname and port only — no additional path segments.</p>
					</div>

					<div>
						<label class="label" for="config-model">Model Name</label>
						<input
							id="config-model"
							type="text"
							bind:value={apiModelName}
							class="input"
							placeholder="gemma3"
						/>
					</div>

					<div>
						<label class="label" for="config-env">Environment</label>
						<input
							id="config-env"
							type="text"
							bind:value={envName}
							class="input"
							placeholder="default"
						/>
						<p class="help-text">Set this to configure API settings outside the TOML file.</p>
					</div>
				</div>
			</section>

			<!-- ═══ Section: Prompt ═══ -->
			<section class="space-y-3">
				<h3 class="section-heading">Prompt</h3>

				<div>
					<label class="label" for="config-prompt">Template</label>
					<select
						id="config-prompt"
						class="input cursor-pointer"
						value={promptName}
						onchange={(e) => {
							promptName = e.currentTarget.value;
						}}
					>
						<option value="">(default)</option>
						{#each $templates.items as t (t.name)}
							<option value={t.name}>{t.name}{t.source === 'builtin' ? ' (built-in)' : ''}</option>
						{/each}
					</select>
					<p class="help-text">
						Prompt template used during captioning. Leave empty for the default template.
					</p>
				</div>
			</section>

			<!-- ═══ Section: Options ═══ -->
			<section class="space-y-3">
				<h3 class="section-heading">Options</h3>

				<div class="grid grid-cols-2 gap-3">
					<!-- Max Tokens -->
					<div>
						<label class="label" for="config-tokens">Max Tokens</label>
						<input
							id="config-tokens"
							type="number"
							value={maxTokens ?? ''}
							min={100}
							max={16384}
							class="input"
							placeholder="512 (default)"
							oninput={(e) => {
								const val = e.currentTarget.value;
								maxTokens = val === '' ? null : Number(val);
							}}
						/>
						<p class="help-text">
							Maximum tokens the model can output. Clear to use default (512).
						</p>
					</div>

					<!-- Image Quality -->
					<div>
						<label class="label" for="config-quality">Image Quality</label>
						<select
							id="config-quality"
							class="input cursor-pointer"
							value={imageQuality ?? ''}
							onchange={(e) => {
								const val = e.currentTarget.value;
								imageQuality = (val === '' ? null : val) as 'auto' | 'high' | 'low' | null;
							}}
						>
							<option value="">Auto (default)</option>
							<option value="auto">Auto</option>
							<option value="high">High</option>
							<option value="low">Low</option>
						</select>
						<p class="help-text">
							Quality of images sent to the model. Higher quality uses more tokens.
						</p>
					</div>

					<!-- Rounds -->
					<div>
						<label class="label" for="config-rounds">Rounds</label>
						<input
							id="config-rounds"
							type="number"
							value={rounds ?? ''}
							min={1}
							max={10}
							class="input"
							placeholder="1 (default)"
							oninput={(e) => {
								const val = e.currentTarget.value;
								rounds = val === '' ? null : Number(val);
							}}
						/>
						<p class="help-text">
							Multiple rounds generate captions, then a final round refines them. Clear to use
							default (1).
						</p>
					</div>
				</div>

				<!-- Checkboxes -->
				<div class="space-y-2">
					<div>
						<div class="flex items-center gap-2">
							<Checkbox id="config-overwrite" bind:checked={overwrite} />
							<label class="cursor-pointer text-sm text-gray-300" for="config-overwrite">
								Overwrite existing captions
							</label>
						</div>
						<p class="help-text ml-6">When disabled, images with existing captions are skipped.</p>
					</div>
					<div>
						<div class="flex items-center gap-2">
							<Checkbox id="config-store-conversation" bind:checked={storeConversation} />
							<label class="cursor-pointer text-sm text-gray-300" for="config-store-conversation">
								Store conversation history
							</label>
						</div>
						<p class="help-text ml-6">
							Let the API provider store conversations for later review. Not recommended for large
							datasets.
						</p>
					</div>
				</div>
			</section>

			<!-- ═══ Section: Reasoning ═══ -->
			<section class="space-y-3">
				<div class="flex items-center gap-2">
					<Checkbox id="config-reasoning" bind:checked={reasoningEnabled} />
					<label class="section-heading cursor-pointer" for="config-reasoning"> Reasoning </label>
				</div>
				<p class="help-text">
					Enables chain-of-thought reasoning for models that support it. Improves output quality at
					the cost of inference time.
				</p>

				{#if reasoningEnabled}
					<div class="grid grid-cols-2 gap-3">
						<div>
							<label class="label" for="config-effort">Thinking Effort</label>
							<select id="config-effort" class="input cursor-pointer" bind:value={reasoningEffort}>
								<option value="low">Low</option>
								<option value="medium">Medium</option>
								<option value="high">High</option>
							</select>
							<p class="help-text">
								How much thinking the model does. Start with low and increase if needed.
							</p>
						</div>
						<div class="flex items-end pb-1">
							<div>
								<div class="flex items-center gap-2">
									<Checkbox id="config-exclude-output" bind:checked={reasoningExcludeOutput} />
									<label class="cursor-pointer text-sm text-gray-300" for="config-exclude-output">
										Exclude from output
									</label>
								</div>
								<p class="help-text ml-6">Hide the reasoning section from the caption output.</p>
							</div>
						</div>
					</div>
				{/if}
			</section>

			<!-- ═══ Section: Preview ═══ -->
			<section class="space-y-3">
				<h3 class="section-heading">Preview</h3>
				{#if dirty}
					<p class="text-xs text-accent">Previewing unsaved changes.</p>
				{:else}
					<p class="text-xs text-gray-500">Current config on disk.</p>
				{/if}
				<div class="max-h-[40vh] min-h-[120px] overflow-y-auto">
					<TomlEditor value={previewContent} editable={false} />
				</div>
			</section>
		{/if}
	</div>

	<!-- Footer: sticky save button -->
	{#if !isLoading && !error}
		<div class="flex-shrink-0 border-t border-border p-4">
			<div class="flex gap-2">
				<button class="btn-secondary flex-1" onclick={loadConfig} disabled={isSaving}>
					<SvgRefresh class="mr-1 inline-block h-4 w-4" />
					Reload
				</button>
				<button class="btn-primary flex-1" onclick={handleSave} disabled={isSaving || !dirty}>
					{isSaving ? 'Saving…' : 'Save Config'}
				</button>
			</div>
		</div>
	{/if}
</div>
