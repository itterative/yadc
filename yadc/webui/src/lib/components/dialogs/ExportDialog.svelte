<script lang="ts">
	import Dialog from '$lib/components/ui/Dialog.svelte';
	import Checkbox from '$lib/components/ui/Checkbox.svelte';
	import SvgClose from '$lib/icons/SvgClose.svelte';
	import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
	import {
		fetchExportBackends,
		fetchDatasetDrafts,
		runExport,
		type ExportBackend,
		type ExportResult
	} from '$lib/stores/configs';
	import { fetchDatasets, type DatasetInfo } from '$lib/stores/datasetImages';
	import { friendlyErrorMessage } from '$lib/api';

	interface Props {
		open: boolean;
		onclose: () => void;
		/** Pre-selected dataset name. */
		datasetName?: string;
		/** Called after a successful export. */
		onexported?: (result: ExportResult) => void;
	}

	let { open, onclose, datasetName = '', onexported }: Props = $props();

	// --- Data ---
	let datasets: DatasetInfo[] = $state([]);
	let backends: ExportBackend[] = $state([]);
	let availableDrafts: string[] = $state([]);
	let isLoading = $state(false);
	let isExporting = $state(false);
	let error: string | null = $state(null);

	// --- Form state ---
	let selectedDataset = $state('');
	let selectedBackend = $state('sd-scripts');
	let selectedFormat = $state('jsonl');
	let source: 'caption' | 'draft' = $state('caption');
	let draftName = $state('');
	let chainedDrafts: string[] = $state([]);
	let chainInput = $state('');
	let outputPath = $state('');
	let append = $state(false);
	let captionExtension = $state('.txt');

	// --- Result ---
	let result: ExportResult | null = $state(null);

	// --- Derived ---
	let currentBackend = $derived(backends.find((b) => b.name === selectedBackend));
	let formats = $derived(currentBackend?.formats ?? ['jsonl']);
	let remainingDrafts = $derived(
		availableDrafts.filter((d) => d !== draftName && !chainedDrafts.includes(d))
	);

	// --- Load data on open ---
	$effect(() => {
		if (open) {
			result = null;
			error = null;
			selectedDataset = datasetName;
			load();
		}
	});

	// When backend changes, reset format to first available
	$effect(() => {
		if (currentBackend && !currentBackend.formats.includes(selectedFormat)) {
			selectedFormat = currentBackend.formats[0];
		}
	});

	// Load available drafts when dataset changes
	$effect(() => {
		chainedDrafts = [];
		if (selectedDataset) {
			loadDrafts(selectedDataset);
		} else {
			availableDrafts = [];
		}
	});

	async function load() {
		isLoading = true;
		error = null;
		try {
			const [ds, be] = await Promise.all([fetchDatasets(), fetchExportBackends()]);
			datasets = ds;
			backends = be;
		} catch (e) {
			error = friendlyErrorMessage(e, 'Failed to load export options');
		} finally {
			isLoading = false;
		}
	}

	async function loadDrafts(dataset: string) {
		try {
			availableDrafts = await fetchDatasetDrafts(dataset);
		} catch {
			availableDrafts = [];
		}
	}

	function addChainedDraft(name: string) {
		if (name && !chainedDrafts.includes(name) && name !== draftName) {
			chainedDrafts = [...chainedDrafts, name];
		}
		chainInput = '';
	}

	function removeChainedDraft(index: number) {
		chainedDrafts = chainedDrafts.filter((_, i) => i !== index);
	}

	function handleChainKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			const val = chainInput.trim();
			if (val) {
				addChainedDraft(val);
			}
		}
	}

	async function handleExport() {
		if (!selectedDataset) {
			error = 'Please select a dataset';
			return;
		}

		isExporting = true;
		error = null;
		result = null;

		try {
			const opts: Record<string, unknown> = {
				dataset: selectedDataset,
				backend: selectedBackend,
				format: selectedFormat,
				source,
				append,
				caption_extension: captionExtension
			};

			if (source === 'draft' && draftName.trim()) {
				opts.draft = draftName.trim();
			}
			if (chainedDrafts.length > 0) {
				opts.with_drafts = chainedDrafts;
			}
			if (outputPath.trim()) {
				opts.output = outputPath.trim();
			}

			const res = await runExport(opts as Parameters<typeof runExport>[0]);
			result = res;
			onexported?.(res);
		} catch (e) {
			error = friendlyErrorMessage(e, 'Export failed');
		} finally {
			isExporting = false;
		}
	}
</script>

<Dialog class="dialog-panel max-h-[85vh] max-w-lg overflow-y-auto" {open} {onclose}>
	<div class="space-y-5 p-6">
		<!-- Header -->
		<div class="dialog-header">
			<h2 class="dialog-title">Export Dataset</h2>
			<button class="btn-close" onclick={onclose}>
				<SvgClose class="h-5 w-5" />
			</button>
		</div>

		{#if error}
			<div class="alert-error">{error}</div>
		{/if}

		{#if result}
			<!-- Success state -->
			<div class="space-y-2 rounded-lg border border-green-700/30 bg-green-900/20 p-4">
				<p class="text-sm font-medium text-green-300">Export complete</p>
				<div class="space-y-1 text-sm text-gray-300">
					<p><span class="text-gray-500">Images exported:</span> {result.count}</p>
					<p><span class="text-gray-500">Format:</span> {result.format}</p>
					<p><span class="text-gray-500">Output:</span> {result.output}</p>
				</div>
				<button class="btn-secondary mt-2" onclick={() => (result = null)}> Export Again </button>
			</div>
		{:else if isLoading}
			<SpinnerBlock class="py-8" />
		{:else}
			<!-- Dataset -->
			<div>
				<label class="label" for="export-dataset">Dataset</label>
				<select id="export-dataset" class="input cursor-pointer" bind:value={selectedDataset}>
					<option value="" disabled>Select dataset</option>
					{#each datasets as ds (ds.name)}
						<option value={ds.name}>{ds.name} ({ds.image_count} images)</option>
					{/each}
				</select>
			</div>

			<!-- Backend & Format -->
			<div class="grid grid-cols-2 gap-3">
				<div>
					<label class="label" for="export-backend">Backend</label>
					<select id="export-backend" class="input cursor-pointer" bind:value={selectedBackend}>
						{#each backends as b (b.name)}
							<option value={b.name}>{b.name}</option>
						{/each}
					</select>
					{#if currentBackend}
						<p class="help-text">{currentBackend.description}</p>
					{/if}
				</div>
				<div>
					<label class="label" for="export-format">Format</label>
					<select id="export-format" class="input cursor-pointer" bind:value={selectedFormat}>
						{#each formats as f (f)}
							<option value={f}>{f}</option>
						{/each}
					</select>
				</div>
			</div>

			<!-- Source -->
			<div>
				<fieldset>
					<legend class="label">Source</legend>
					<div class="flex gap-4">
						<label class="flex cursor-pointer items-center gap-2 text-sm text-gray-200">
							<input type="radio" name="export-source" value="caption" bind:group={source} />
							Captions
						</label>
						<label class="flex cursor-pointer items-center gap-2 text-sm text-gray-200">
							<input type="radio" name="export-source" value="draft" bind:group={source} />
							Draft
						</label>
					</div>
				</fieldset>
				{#if source === 'draft'}
					<div class="mt-2">
						<input type="text" bind:value={draftName} class="input" placeholder="Draft name" />
						{#if availableDrafts.length > 0}
							<div class="mt-2 flex flex-wrap gap-1.5">
								{#each availableDrafts as d (d)}
									{@const active = draftName === d}
									<button
										class="cursor-pointer rounded-md px-2 py-0.5 text-xs font-medium transition-colors {active
											? 'border border-accent/40 bg-accent/25 text-accent'
											: 'border border-border bg-bg text-gray-400 hover:border-gray-500 hover:text-gray-200'}"
										onclick={() => (draftName = active ? '' : d)}
									>
										{d}
									</button>
								{/each}
							</div>
						{/if}
					</div>
				{/if}
			</div>

			<!-- Chain drafts -->
			<div>
				<label class="label" for="chain-draft-input">
					Chain Drafts
					<span class="ml-1 text-gray-600">(appended in order after source)</span>
				</label>
				{#if chainedDrafts.length > 0}
					<div class="mb-2 flex flex-wrap gap-1.5">
						{#each chainedDrafts as name, i (i)}
							<span
								class="inline-flex items-center gap-1 rounded-md border border-accent/25 bg-accent/15 px-2 py-0.5 text-xs font-medium text-accent"
							>
								{name}
								<button
									class="cursor-pointer transition-colors hover:text-white"
									onclick={() => removeChainedDraft(i)}
								>
									<SvgClose class="h-3 w-3" />
								</button>
							</span>
						{/each}
					</div>
				{/if}
				<input
					id="chain-draft-input"
					type="text"
					bind:value={chainInput}
					onkeydown={handleChainKeydown}
					class="input"
					placeholder="Type draft name, press Enter to add"
				/>
				{#if remainingDrafts.length > 0}
					<div class="mt-2 flex flex-wrap gap-1.5">
						{#each remainingDrafts as d (d)}
							<button
								class="cursor-pointer rounded-md border border-border bg-bg px-2 py-0.5 text-xs font-medium text-gray-400 transition-colors hover:border-gray-500 hover:text-gray-200"
								onclick={() => addChainedDraft(d)}
							>
								+ {d}
							</button>
						{/each}
					</div>
				{/if}
			</div>

			<!-- Output path -->
			<div>
				<label class="label" for="export-output">
					Output Path
					<span class="ml-1 text-gray-600">(optional — auto-detected if empty)</span>
				</label>
				<input
					id="export-output"
					type="text"
					bind:value={outputPath}
					class="input"
					placeholder="e.g. /data/training/metadata.jsonl"
				/>
			</div>

			<!-- Options -->
			<div class="flex items-center gap-6">
				<div class="flex items-center gap-2">
					<Checkbox id="export-append" bind:checked={append} />
					<label class="cursor-pointer text-sm text-gray-300" for="export-append">Append</label>
				</div>
				<div class="flex items-center gap-2">
					<label class="text-sm text-gray-400" for="export-ext">Extension</label>
					<input
						id="export-ext"
						type="text"
						bind:value={captionExtension}
						class="input-sm w-20 px-2 py-1"
					/>
				</div>
			</div>

			<!-- Footer -->
			<div class="btn-bar border-t border-border">
				<button class="btn-secondary" onclick={onclose}> Cancel </button>
				<button
					class="btn-primary"
					onclick={handleExport}
					disabled={isExporting || !selectedDataset}
				>
					{isExporting ? 'Exporting…' : 'Export'}
				</button>
			</div>
		{/if}
	</div>
</Dialog>
