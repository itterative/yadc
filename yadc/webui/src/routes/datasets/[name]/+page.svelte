<script lang="ts">
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';
	import { get } from 'svelte/store';
	import DatasetBrowser from '$lib/components/dataset/DatasetBrowser.svelte';
	import SidePanel from './SidePanel.svelte';
	import SvgChevronLeft from '$lib/icons/SvgChevronLeft.svelte';
	import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
	import type { CaptionOptions } from '$lib/stores/captionOptions';
	import {
		captioningStatus,
		pendingDatasetChanges,
		clearPendingDatasetChange,
		registerJobId,
		resumptionFailed,
		clearResumptionFailed,
		setCaptioningStatus
	} from '$lib/stores/events';
	import { toast } from '$lib/stores/toasts';
	import { API_BASE, apiErrorMessage, friendlyErrorMessage } from '$lib/api';
	import {
		fetchDatasets,
		fetchImages,
		fetchCaptioningStatus,
		startCaptioning,
		captionSingleImage,
		type DatasetInfo,
		type ImageInfo
	} from '$lib/stores/datasetImages';

	let datasetName = $derived($page.params.name ?? '');

	let datasets: DatasetInfo[] = $state([]);
	let currentDataset: DatasetInfo | null = $state(null);
	let focusedItem: ImageInfo | null = $state(null);

	let images: ImageInfo[] = $state([]);
	let isLoading = $state(true);
	let isLoadingMore = $state(false);
	let hasMore = $state(true);
	let error: string | null = $state(null);

	let nextToken: string | null = null;
	const PAGE_SIZE = 50;

	// --- Mobile panel state ---
	let panelOpen = $state(false);

	// --- Captioning state ---

	// NOTE: Grid tiles are NOT updated live during captioning — the SSE events
	// (CaptioningStatusEvent) only carry aggregate counts (processed/total/errors),
	// not individual image IDs. A full reload happens when captioning finishes
	// (see handleCaptioningDone). For live per-tile updates, the backend would
	// need to emit per-image events (e.g. CaptionedImageEvent with image_id).

	let captionDoneFired = $state(false);

	// Derive captioning state from the global SSE store
	let isCaptioning = $derived(
		$captioningStatus?.dataset_name === datasetName &&
			($captioningStatus.status === 'running' || $captioningStatus.status === 'stopping')
	);

	let isStopping = $derived(
		$captioningStatus?.dataset_name === datasetName && $captioningStatus.status === 'stopping'
	);

	let captionPct = $derived(
		$captioningStatus?.total > 0
			? Math.round(($captioningStatus.processed / $captioningStatus.total) * 100)
			: 0
	);

	async function handleStopCaptioning() {
		try {
			const res = await fetch(
				`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`,
				{ method: 'DELETE' }
			);
			if (!res.ok) {
				toast.error(`Failed to stop captioning: ${await apiErrorMessage(res)}`);
			}
		} catch {
			toast.error('Failed to stop captioning: request failed');
		}
	}

	// Register job_id from incoming status events so dataset_changed events
	// from our own captioning are suppressed
	$effect(() => {
		const s = $captioningStatus;
		if (
			s?.dataset_name === datasetName &&
			s.job_id &&
			(s.status === 'running' || s.status === 'stopping')
		) {
			registerJobId(s.job_id);
		}
	});

	// Detect when captioning finishes (transition from active → terminal)
	$effect(() => {
		if (!isCaptioning && !captionDoneFired) {
			captionDoneFired = true;
			handleCaptioningDone();
		}
		if (isCaptioning) {
			captionDoneFired = false;
		}
	});

	// Live caption options from CaptionSettings (updated reactively as settings change)
	let captionOptions: CaptionOptions = $state({});

	// --- Filesystem watcher state ---

	let hasPendingChanges = $state(false);

	$effect(() => {
		const name = datasetName;
		return pendingDatasetChanges.subscribe((set) => {
			hasPendingChanges = set.has(name);
		});
	});

	function handleDismissResumptionFailed() {
		clearResumptionFailed();
	}

	async function loadInitial(name: string) {
		nextToken = null;
		isLoading = true;
		error = null;

		try {
			const page = await fetchImages(name, { limit: PAGE_SIZE });
			images = page.images;
			nextToken = page.next_token;
			hasMore = page.next_token !== null;
		} catch (e) {
			error = friendlyErrorMessage(e, 'Failed to load images');
			toast.error(`Failed to load images: ${error}`);
		} finally {
			isLoading = false;
		}
	}

	async function loadMore() {
		if (!datasetName) {
			return;
		}
		if (isLoadingMore || !hasMore || nextToken === null) {
			return;
		}
		isLoadingMore = true;

		try {
			const page = await fetchImages(datasetName, {
				limit: PAGE_SIZE,
				afterId: parseInt(nextToken)
			});
			images = [...images, ...page.images];
			nextToken = page.next_token;
			hasMore = page.next_token !== null;
		} catch (e) {
			error = friendlyErrorMessage(e, 'Failed to load more images');
		} finally {
			isLoadingMore = false;
		}
	}

	// React to dataset name changes
	$effect(() => {
		const name = datasetName;
		if (!browser || !name) {
			return;
		}
		loadInitial(name);
		// Poll current captioning status to handle mid-captioning page loads.
		// Only seed the store if it is idle for this dataset so we do not
		// clobber a more recent SSE event.
		(async () => {
			try {
				const status = await fetchCaptioningStatus(name);
				const current = get(captioningStatus);
				if (
					status.dataset_name === name &&
					(current.status === 'idle' || current.dataset_name !== name)
				) {
					setCaptioningStatus(status);
				}
			} catch {
				/* ignore — SSE will eventually provide status */
			}
		})();
	});

	// Find the dataset info from the list
	$effect(() => {
		if (datasets.length > 0 && datasetName) {
			currentDataset = datasets.find((d) => d.name === datasetName) ?? null;
		}
	});

	onMount(() => {
		if (!browser) {
			return;
		}
		(async () => {
			try {
				datasets = await fetchDatasets();
			} catch {
				datasets = [];
			}
		})();
	});

	// --- Side panel state ---

	let panelTab: 'caption' | 'details' | 'config' = $state('caption');

	function handleItemClick(item: ImageInfo) {
		focusedItem = item;
		panelTab = 'details';
		panelOpen = true;
	}

	function handlePanelClose() {
		focusedItem = null;
		panelTab = 'caption';
		panelOpen = false;
	}

	function handleCaptionUpdated(imageId: number, _caption: string) {
		void _caption;
		images = images.map((img) => (img.id === imageId ? { ...img, has_caption: true } : img));
	}

	async function handleCaptionImage(imageId: number) {
		try {
			const result = await captionSingleImage(
				datasetName,
				imageId,
				captionOptions as Record<string, unknown>
			);
			registerJobId(result.job_id);
			if (result.caption) {
				images = images.map((img) => (img.id === imageId ? { ...img, has_caption: true } : img));
			}
			return result.caption;
		} catch (e) {
			throw e instanceof Error ? e : new Error('Failed to caption image');
		}
	}

	async function handleStartCaptioning(options: CaptionOptions) {
		captionOptions = options;
		panelOpen = false;
		try {
			const info = await startCaptioning(datasetName, options as Record<string, unknown>);
			registerJobId(info.job_id);
		} catch (e) {
			toast.error(friendlyErrorMessage(e, 'Failed to start captioning'));
		}
	}

	function handleCaptioningDone() {
		// Refresh everything — caption counts and image states may have changed
		if (browser && datasetName) {
			loadInitial(datasetName);
			(async () => {
				try {
					datasets = await fetchDatasets();
				} catch {
					/* ignore */
				}
			})();
		}
	}

	function handleRefreshFromWatcher() {
		if (!browser || !datasetName) {
			return;
		}
		clearPendingDatasetChange(datasetName);
		loadInitial(datasetName);
		(async () => {
			try {
				datasets = await fetchDatasets();
			} catch {
				/* ignore */
			}
		})();
	}
</script>

<svelte:head>
	<title>{datasetName} — yadc</title>
</svelte:head>

<div class="flex h-full flex-col gap-4">
	<!-- Header -->
	<div class="flex shrink-0 items-center gap-4">
		<a href="#/" class="-ml-2 p-2 text-gray-400 transition-colors hover:text-white">
			<SvgChevronLeft class="h-6 w-6" />
		</a>
		<div class="min-w-0 flex-1">
			<h1 class="text-xl font-bold text-white">{datasetName}</h1>
			<div class="mt-0.5 flex min-h-7 w-full items-center">
				{#if isCaptioning}
					<div class="flex w-full items-center gap-2">
						{#if isStopping}
							<SvgSpinner class="h-3.5 w-3.5 shrink-0 animate-spin text-yellow-400" />
							<span class="text-sm text-yellow-300">Stopping…</span>
						{:else}
							<SvgSpinner class="h-3.5 w-3.5 shrink-0 animate-spin text-accent" />
							<span class="text-sm text-gray-400">
								Captioning… {$captioningStatus.processed}/{$captioningStatus.total}
							</span>
							<div class="h-1.5 max-w-32 flex-1 overflow-hidden rounded-full bg-bg">
								<div
									class="h-full rounded-full bg-accent transition-all duration-300 ease-out"
									style:width="{captionPct}%"
								></div>
							</div>
							<span class="text-xs text-gray-500">{captionPct}%</span>
							<button
								class="cursor-pointer rounded-md border border-error/30 bg-error/20 px-2 py-0.5 text-xs text-error transition-colors hover:bg-error/30 disabled:opacity-50"
								onclick={handleStopCaptioning}
							>
								Stop
							</button>
						{/if}
					</div>
				{:else if currentDataset}
					<p class="text-sm text-gray-400">
						{currentDataset.image_count} images · {currentDataset.has_caption} captioned · {currentDataset.has_toml}
						with TOML
					</p>
				{/if}
			</div>
		</div>
	</div>

	<!-- Filesystem change notification -->
	{#if hasPendingChanges}
		<div
			class="flex items-center justify-between rounded-lg border border-blue-700/50 bg-blue-900/50 px-4 py-2"
		>
			<span class="text-sm text-blue-200">Dataset files have changed</span>
			<button
				class="cursor-pointer rounded-lg border border-blue-500/50 bg-blue-600/40 px-3 py-1.5 text-xs text-blue-200 transition-colors hover:bg-blue-600/60"
				onclick={handleRefreshFromWatcher}
			>
				Refresh
			</button>
		</div>
	{/if}

	<!-- SSE resumption failure notification -->
	{#if $resumptionFailed}
		<div
			class="flex items-center justify-between gap-4 rounded-lg border border-yellow-700/50 bg-yellow-900/50 px-4 py-2"
		>
			<span class="text-sm text-yellow-200"
				>Some events may have been missed due to a disconnected event stream.</span
			>
			<div class="flex shrink-0 items-center gap-2">
				<button
					class="cursor-pointer rounded-lg border border-yellow-500/50 bg-yellow-600/40 px-3 py-1.5 text-xs text-yellow-200 transition-colors hover:bg-yellow-600/60"
					onclick={() => {
						handleRefreshFromWatcher();
						handleDismissResumptionFailed();
					}}
				>
					Refresh
				</button>
				<button
					class="cursor-pointer rounded-lg border border-yellow-500/30 bg-transparent px-2 py-1.5 text-xs text-yellow-300 transition-colors hover:bg-yellow-600/20"
					onclick={handleDismissResumptionFailed}
				>
					Dismiss
				</button>
			</div>
		</div>
	{/if}

	<!-- Content: flex row with grid + side panel -->

	<div class="relative flex min-h-0 flex-1 gap-4">
		{#if error}
			<p class="text-error">Error: {error}</p>
		{:else}
			<!-- Image grid (full width on mobile, flex-1 on desktop) -->
			<div class="min-w-0 flex-1 overflow-y-auto">
				<DatasetBrowser
					class="mx-auto"
					{datasetName}
					items={images}
					{isLoading}
					{isLoadingMore}
					selectedId={focusedItem?.id ?? null}
					onclick={handleItemClick}
					onendreached={loadMore}
				/>

				{#if !isLoading && images.length === 0}
					<div class="empty-state">
						<p class="text-lg">No images found</p>
						<p class="mt-1 text-sm">This dataset may be empty or not yet scanned.</p>
					</div>
				{/if}
			</div>
		{/if}

		<SidePanel
			{datasetName}
			{focusedItem}
			bind:activeTab={panelTab}
			bind:open={panelOpen}
			bind:captionOptions
			onstartcaptioning={handleStartCaptioning}
			onpanelclose={handlePanelClose}
			oncaptionupdated={handleCaptionUpdated}
			oncaptionimage={handleCaptionImage}
			onconfigsaved={handleRefreshFromWatcher}
		/>
	</div>
</div>
