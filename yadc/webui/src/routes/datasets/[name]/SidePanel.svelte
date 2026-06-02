<script lang="ts">
	import CaptionSettings from './CaptionSettings.svelte';
	import DatasetConfig from './DatasetConfig.svelte';
	import ImageDetail from '$lib/components/dataset/ImageDetail.svelte';
	import type { CaptionOptions } from '$lib/stores/captionOptions';
	import type { ImageInfo } from '$lib/stores/datasetImages';
	import SvgClose from '$lib/icons/SvgClose.svelte';
	import SvgMenuLeft from '$lib/icons/SvgMenuLeft.svelte';

	type PanelTab = 'caption' | 'details' | 'config';

	interface Props {
		datasetName: string;
		focusedItem: ImageInfo | null;
		panelTab?: PanelTab;
		open?: boolean;
		captionOptions?: CaptionOptions;
		onstartcaptioning?: (options: CaptionOptions) => void;
		onpanelclose?: () => void;
		oncaptionupdated?: (imageId: number, caption: string) => void;
		oncaptionimage?: (imageId: number) => Promise<string>;
		onconfigsaved?: () => void;
	}

	let {
		datasetName,
		focusedItem,
		panelTab = $bindable('caption'),
		open = $bindable(false),
		captionOptions = $bindable(),
		onstartcaptioning,
		onpanelclose,
		oncaptionupdated,
		oncaptionimage,
		onconfigsaved
	}: Props = $props();
</script>

<!-- Mobile: backdrop -->
{#if open}
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="fixed inset-0 z-30 bg-black/50 transition-opacity lg:hidden"
		onclick={() => {
			open = false;
			onpanelclose?.();
		}}
	></div>
{/if}

<!-- Side panel: desktop = inline, mobile = slide-over drawer -->
<div
	class="side-panel
    fixed inset-y-0 right-0 z-40 w-[80vw] max-w-[480px] shadow-2xl transition-transform duration-300
    lg:static lg:z-auto lg:w-[480px] lg:max-w-none lg:flex-shrink-0 lg:shadow-none lg:transition-none
    {open ? '' : 'translate-x-full'} lg:translate-x-0"
>
	<div
		class="flex h-full flex-col overflow-hidden border-l border-border bg-surface lg:rounded-xl lg:border"
	>
		<!-- Tab bar -->
		<div class="flex flex-shrink-0 items-center border-b border-border">
			<button
				class="cursor-pointer border-b-2 px-4 py-2.5 text-sm font-medium transition-colors {panelTab ===
				'caption'
					? 'border-accent text-white'
					: 'border-transparent text-gray-400 hover:text-gray-200'}"
				onclick={() => (panelTab = 'caption')}
			>
				Caption
			</button>
			<button
				class="cursor-pointer border-b-2 px-4 py-2.5 text-sm font-medium transition-colors {panelTab ===
				'details'
					? 'border-accent text-white'
					: 'border-transparent text-gray-400 hover:text-gray-200'}"
				onclick={() => (panelTab = 'details')}
			>
				Details
			</button>
			<button
				class="cursor-pointer border-b-2 px-4 py-2.5 text-sm font-medium transition-colors {panelTab ===
				'config'
					? 'border-accent text-white'
					: 'border-transparent text-gray-400 hover:text-gray-200'}"
				onclick={() => (panelTab = 'config')}
			>
				Config
			</button>

			<button
				class="mr-2 ml-auto cursor-pointer p-1 text-gray-400 transition-colors hover:text-white lg:hidden"
				onclick={() => {
					open = false;
					onpanelclose?.();
				}}
				title="Close panel"
				aria-label="Close panel"
			>
				<SvgClose class="h-5 w-5" />
			</button>
		</div>

		<!-- Tab content: always render both to preserve component state across tab switches -->
		<div class="min-h-0 flex-1 overflow-y-auto">
			<div class={panelTab === 'caption' ? '' : 'hidden'}>
				<CaptionSettings
					{datasetName}
					bind:currentOptions={captionOptions}
					onstart={onstartcaptioning}
					onclose={() => {
						open = false;
					}}
				/>
			</div>
			<div class={panelTab === 'details' ? '' : 'hidden'}>
				{#if focusedItem !== null}
					<ImageDetail {datasetName} item={focusedItem} {oncaptionupdated} {oncaptionimage} />
				{:else}
					<div class="flex h-full items-center justify-center text-sm text-gray-500">
						Select an image to view details
					</div>
				{/if}
			</div>
			<div class={panelTab === 'config' ? '' : 'hidden'}>
				<DatasetConfig {datasetName} onsaved={onconfigsaved} />
			</div>
		</div>
	</div>
</div>

<!-- Mobile: floating toggle button -->
<button
	class="fixed right-6 bottom-6 z-20 flex h-16 w-16 cursor-pointer items-center justify-center rounded-full
         bg-accent text-white shadow-lg transition-colors hover:bg-accent-hover lg:hidden"
	onclick={() => (open = !open)}
	title="Toggle panel"
>
	<SvgMenuLeft class="h-6 w-6" />
</button>
