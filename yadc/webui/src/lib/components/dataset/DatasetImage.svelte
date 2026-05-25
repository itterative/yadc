<script lang="ts">
	import type { ImageInfo } from '$lib/stores/datasetImages';

	interface Props {
		class?: string;
		datasetName: string;
		item: ImageInfo;
		selected?: boolean;
		onclick: (item: ImageInfo) => void;
	}

	let { class: klazz = '', datasetName, item, selected = false, onclick }: Props = $props();

	let media: HTMLImageElement | null = $state(null);
	let loading: boolean = $state(true);

	let thumbnailSrc = $derived(
		`/api/datasets/${encodeURIComponent(datasetName)}/images/${item.id}/thumbnail?size=512`
	);

	let aspectRatio = $derived.by(() => {
		if (!item.width || !item.height) {
			return undefined;
		}
		return `${item.width} / ${item.height}`;
	});

	$effect(() => {
		const setLoadingToFalse = () => (loading = false);
		media?.addEventListener('load', setLoadingToFalse);
		return () => media?.removeEventListener('load', setLoadingToFalse);
	});
</script>

<button
	class={klazz}
	onclick={() => onclick(item)}
	class:has-caption={item.has_caption}
	class:has-toml={item.has_toml}
	class:selected
>
	<img
		src={thumbnailSrc}
		alt={item.file_name}
		loading="lazy"
		class="block w-full bg-gray-500"
		class:animate-pulse={loading}
		width={item.width || undefined}
		height={item.height || undefined}
		style:aspect-ratio={aspectRatio}
		bind:this={media}
	/>

	{#if item.has_caption}
		<span
			class="badge-sm pointer-events-none absolute top-1 right-1 bg-success text-black"
			title="Has caption">✓</span
		>
	{/if}

	{#if item.has_toml}
		<span
			class="badge-sm pointer-events-none absolute top-1 right-6 bg-accent text-black"
			title="Has TOML extras">T</span
		>
	{/if}

	{#if item.draft_names.length > 0}
		<span
			class="badge-sm pointer-events-none absolute top-1 left-1 bg-error text-white"
			title="{item.draft_names.length} draft(s)">D</span
		>
	{/if}
</button>

<style>
	/* .has-caption — indicator applied via has-caption class name, styling via border in parent */

	.selected {
		outline: 2px solid var(--color-accent);
		outline-offset: -2px;
	}
</style>
