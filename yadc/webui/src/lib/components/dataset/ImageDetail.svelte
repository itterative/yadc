<script lang="ts">
	import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
	import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
	import PromptPreview from '$lib/components/ui/PromptPreview.svelte';
	import {
		fetchCaption,
		mediaUrl,
		updateCaption,
		updateExtras,
		type CaptionData,
		type ImageInfo
	} from '$lib/stores/datasetImages';
	import SvgSpinner from '$lib/icons/SvgSpinner.svelte';

	interface Props {
		datasetName: string;
		item: ImageInfo | null;
		oncaptionupdated?: (imageId: number, caption: string) => void;
		oncaptionimage?: (imageId: number) => Promise<string>;
	}

	let { datasetName, item, oncaptionupdated, oncaptionimage }: Props = $props();

	let captionData: CaptionData | null = $state(null);
	let isLoadingCaption = $state(false);
	let captionError: string | null = $state(null);
	let isEditing = $state(false);
	let editCaption = $state('');
	let isSaving = $state(false);

	// Single-image captioning state
	let isCaptioning = $state(false);
	let captioningError: string | null = $state(null);

	// TOML extras editing state
	let isEditingExtras = $state(false);
	let editExtrasRaw = $state('');

	let imgElement: HTMLImageElement | null = $state(null);

	// Load caption when item changes
	$effect(() => {
		if (item === null) {
			captionData = null;
			captionError = null;
			isEditing = false;
			isEditingExtras = false;
			return;
		}

		let cancelled = false;
		isLoadingCaption = true;
		captionError = null;
		captionData = null;
		isEditing = false;
		isEditingExtras = false;

		(async () => {
			try {
				const data = await fetchCaption(datasetName, item.id);
				if (cancelled) {
					return;
				}
				captionData = data;
				editCaption = data.caption || '';
			} catch (e) {
				if (cancelled) {
					return;
				}
				captionError = e instanceof Error ? e.message : 'Failed to load caption';
			} finally {
				if (!cancelled) {
					isLoadingCaption = false;
				}
			}
		})();

		return () => {
			cancelled = true;
		};
	});

	let imgSrc = $derived(item !== null ? mediaUrl(datasetName, item.id) : '');

	async function handleSave() {
		if (item === null || !isEditing) {
			return;
		}
		isSaving = true;
		try {
			await updateCaption(datasetName, item.id, editCaption);
			captionData = { ...captionData!, caption: editCaption };
			isEditing = false;
			oncaptionupdated?.(item.id, editCaption);
		} catch (e) {
			captionError = e instanceof Error ? e.message : 'Failed to save caption';
		} finally {
			isSaving = false;
		}
	}

	function handleCancelEdit() {
		isEditing = false;
		editCaption = captionData?.caption || '';
	}

	async function handleCaptionImage() {
		if (item === null || !oncaptionimage) {
			return;
		}
		isCaptioning = true;
		captioningError = null;
		try {
			const caption = await oncaptionimage(item.id);
			if (caption) {
				captionData = { ...captionData!, caption };
				oncaptionupdated?.(item.id, caption);
			}
		} catch (e) {
			captioningError = e instanceof Error ? e.message : 'Failed to caption image';
		} finally {
			isCaptioning = false;
		}
	}

	let isSavingExtras = $state(false);

	async function handleSaveExtras() {
		if (item === null || !isEditingExtras) {
			return;
		}
		isSavingExtras = true;
		try {
			await updateExtras(datasetName, item.id, editExtrasRaw);
			captionData = { ...captionData!, extras_raw: editExtrasRaw };
			isEditingExtras = false;
		} catch (e) {
			captionError = e instanceof Error ? e.message : 'Failed to save extras';
		} finally {
			isSavingExtras = false;
		}
	}

	function handleCancelEditExtras() {
		isEditingExtras = false;
		editExtrasRaw = captionData?.extras_raw || '';
	}

	function autosize(el: HTMLTextAreaElement) {
		function resize() {
			el.style.height = 'auto';
			el.style.height = Math.min(el.scrollHeight, 360) + 'px';
		}
		resize();
		return { update: resize };
	}
</script>

{#if item !== null}
	<div class="flex h-full flex-col">
		<!-- Scrollable content -->
		<div class="flex-1 space-y-4 overflow-y-auto p-4">
			<!-- Filename -->
			<h2 class="dialog-title truncate text-base">{item.file_name}</h2>
			<p class="-mt-3 truncate text-xs text-gray-500">{item.path}</p>
			<!-- Image -->
			<div class="flex flex-shrink-0 items-start justify-center">
				<img
					src={imgSrc}
					alt={item.file_name}
					class="max-h-[50vh] w-auto rounded-lg bg-gray-500"
					width={item.width || undefined}
					height={item.height || undefined}
					bind:this={imgElement}
				/>
			</div>

			<!-- Dimensions & badges -->
			<div class="flex flex-wrap items-center gap-3">
				{#if item.width && item.height}
					<span class="text-sm text-gray-400">{item.width}×{item.height}</span>
				{/if}
				{#if item.has_caption}
					<span class="badge-success rounded-full px-2 py-1">Captioned</span>
				{:else}
					<span class="badge-muted rounded-full px-2 py-1">No caption</span>
				{/if}
				{#if item.has_toml}
					<span class="badge-accent rounded-full px-2 py-1">TOML</span>
				{/if}
				{#if item.draft_names.length > 0}
					<span class="badge-error rounded-full px-2 py-1">
						{item.draft_names.length} draft{item.draft_names.length !== 1 ? 's' : ''}
					</span>
				{/if}
			</div>

			<!-- Caption -->
			<div>
				<div class="mb-2 flex items-center justify-between">
					<h3 class="text-sm font-medium text-gray-300">Caption</h3>
					{#if captionData && !isEditing && !isCaptioning}
						<div class="flex items-center gap-3">
							{#if oncaptionimage}
								<button
									class="cursor-pointer text-xs text-accent hover:text-accent-hover"
									onclick={handleCaptionImage}
								>
									Caption
								</button>
							{/if}
							<button
								class="cursor-pointer text-xs text-accent hover:text-accent-hover"
								onclick={() => {
									editCaption = captionData?.caption || '';
									isEditing = true;
								}}
							>
								Edit
							</button>
						</div>
					{/if}
				</div>

				{#if isLoadingCaption}
					<SpinnerBlock size="h-4 w-4" label="Loading caption..." />
				{:else if captionError}
					<p class="text-sm text-error">{captionError}</p>
				{:else}
					<div class="relative">
						{#if isCaptioning}
							<div
								class="absolute inset-0 z-10 flex items-center justify-center gap-2 rounded-lg bg-black/60 text-sm text-gray-300"
							>
								<SvgSpinner class="h-4 w-4 animate-spin" />
								<span>Generating caption…</span>
							</div>
						{:else if captioningError}
							<div
								class="absolute inset-x-0 top-0 z-10 rounded-t-lg bg-error/90 px-3 py-1.5 text-center text-sm text-white"
							>
								{captioningError}
							</div>
						{/if}
						{#if isEditing}
							<div class="space-y-2">
								<textarea
									bind:value={editCaption}
									class="max-h-60 w-full resize-none rounded-lg bg-gray-800 p-3 font-mono text-sm whitespace-pre-wrap text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
									placeholder="Enter caption..."
									oninput={(e) => {
										const el = e.currentTarget;
										el.style.height = 'auto';
										el.style.height = Math.min(el.scrollHeight, 360) + 'px';
									}}
									use:autosize
								></textarea>
								<div class="btn-bar">
									<button
										class="btn-secondary px-3 py-1.5"
										onclick={handleCancelEdit}
										disabled={isSaving}
									>
										Cancel
									</button>
									<button class="btn-primary px-3 py-1.5" onclick={handleSave} disabled={isSaving}>
										{isSaving ? 'Saving...' : 'Save'}
									</button>
								</div>
							</div>
						{:else if captionData}
							{#if captionData.caption}
								<pre
									class="max-h-60 overflow-y-auto rounded-lg bg-gray-800 p-3 text-sm whitespace-pre-wrap text-gray-200">{captionData.caption}</pre>
							{:else}
								<pre
									class="rounded-lg bg-gray-800 p-3 text-sm whitespace-pre-wrap text-gray-500 italic">No caption</pre>
							{/if}
						{/if}
					</div>
				{/if}
			</div>

			<!-- TOML extras -->
			{#if captionData?.extras_raw}
				<div>
					<div class="mb-2 flex items-center justify-between">
						<h3 class="text-sm font-medium text-gray-300">TOML Extras</h3>
						{#if !isEditingExtras}
							<button
								class="cursor-pointer text-xs text-accent hover:text-accent-hover"
								onclick={() => {
									editExtrasRaw = captionData?.extras_raw || '';
									isEditingExtras = true;
								}}
							>
								Edit
							</button>
						{:else}
							<div class="flex gap-2">
								<button
									class="btn-secondary px-2 py-1 text-xs"
									onclick={handleCancelEditExtras}
									disabled={isSavingExtras}
								>
									Cancel
								</button>
								<button
									class="btn-primary px-2 py-1 text-xs"
									onclick={handleSaveExtras}
									disabled={isSavingExtras}
								>
									{isSavingExtras ? 'Saving...' : 'Save'}
								</button>
							</div>
						{/if}
					</div>
					<TomlEditor
						class="text-sm"
						value={isEditingExtras ? editExtrasRaw : captionData.extras_raw}
						editable={isEditingExtras}
						onchange={(v) => (editExtrasRaw = v)}
					/>
				</div>
			{:else if captionData?.extras && Object.keys(captionData.extras).length > 0}
				<div>
					<h3 class="mb-2 text-sm font-medium text-gray-300">TOML Extras</h3>
					<pre
						class="overflow-x-auto rounded-lg bg-gray-800 p-3 text-xs text-gray-400">{JSON.stringify(
							captionData.extras,
							null,
							2
						)}</pre>
				</div>
			{/if}

			<!-- Drafts -->
			{#if captionData?.drafts && Object.keys(captionData.drafts).length > 0}
				<div>
					<h3 class="mb-2 text-sm font-medium text-gray-300">Drafts</h3>
					<div class="space-y-2">
						{#each Object.entries(captionData.drafts) as [name, text] (name)}
							<div class="rounded-lg bg-gray-800 p-3">
								<p class="mb-1 text-xs font-medium text-accent">{name}</p>
								<pre
									class="max-h-40 overflow-y-auto font-mono text-sm whitespace-pre-wrap text-gray-200">{text}</pre>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Prompt Preview -->
			<PromptPreview {datasetName} imageId={item.id} />
		</div>
	</div>
{/if}
