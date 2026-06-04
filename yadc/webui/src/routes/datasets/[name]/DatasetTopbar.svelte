<script lang="ts">
    import type { DatasetInfo } from '$lib/stores/dataset';
    import { captioningStatus, captionTimingRing } from '$lib/stores/events';
    import { formatEta } from '$lib/format';
    import Topbar from '$lib/components/ui/Topbar.svelte';
    import SvgChevronLeft from '$lib/icons/SvgChevronLeft.svelte';
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import SvgUpload from '$lib/icons/SvgUpload.svelte';

    interface Props {
        /** The current dataset name (shown as the page title). */
        datasetName: string;
        /** The current dataset info (for the source badge and stats). */
        currentDataset: DatasetInfo | null;
        /** True when batch captioning is active for this dataset. */
        isBatchCaptioning: boolean;
        /** True when a stop is in progress. */
        isStopping: boolean;
        /** Captioning progress percentage (0-100). */
        captionPct: number;
        /** Number of images processed so far. */
        captionProcessed: number;
        /** Total number of images to caption. */
        captionTotal: number;
        /** True when a manual refresh is in progress. */
        isRefreshing: boolean;
        /** True when uploads are allowed (managed dataset + no batch captioning). */
        canUpload: boolean;
        /** Called when the user clicks the refresh button. */
        onrefresh: () => void;
        /** Called when the user clicks the upload button. */
        onupload: () => void;
    }

    let {
        datasetName,
        currentDataset,
        isBatchCaptioning,
        isStopping,
        captionPct,
        captionProcessed,
        captionTotal,
        isRefreshing,
        canUpload,
        onrefresh,
        onupload
    }: Props = $props();

    function _estimateRemainingSeconds(
        status: import('$lib/stores/events').CaptioningStatus,
        ring: Record<string, number[]>
    ): number | null {
        if (status.status !== 'running' && status.status !== 'stopping') {
            return null;
        }
        if (!status.api_url || !status.api_model_name || status.total <= status.processed) {
            return null;
        }
        const key = `${status.api_url}#${status.api_model_name}`;
        const samples = ring[key];
        if (!samples || samples.length === 0) {
            return null;
        }
        const avgMs = samples.reduce((a, b) => a + b, 0) / samples.length;
        const remaining = status.total - status.processed;
        return Math.round((remaining * avgMs) / 1000);
    }

    let estimatedRemainingSeconds = $derived(
        _estimateRemainingSeconds($captioningStatus, $captionTimingRing)
    );
</script>

<Topbar>
    <a href="#/" class="-ml-2 hidden p-2 text-gray-400 transition-colors hover:text-white md:flex">
        <SvgChevronLeft class="h-6 w-6" />
    </a>
    <div class="relative min-w-0 flex-1">
        <div class="flex items-center gap-3">
            <h1 class="overflow-hidden text-xl font-bold text-nowrap text-ellipsis text-white">
                {datasetName}
            </h1>
            {#if currentDataset}
                {#if currentDataset.source === 'upload'}
                    <span class="badge-muted badge-sm shrink-0">Managed</span>
                {:else}
                    <span class="badge-muted badge-sm shrink-0">External</span>
                {/if}
            {/if}
            {#if isBatchCaptioning}
                <SvgSpinner
                    class="h-4 w-4 shrink-0 animate-spin {isStopping
                        ? 'text-yellow-400'
                        : 'text-accent'}"
                />
            {/if}
        </div>
        <div class="relative mt-0.5 flex items-center">
            {#if isBatchCaptioning}
                {#if isStopping}
                    <span class="text-sm text-yellow-300">Stopping…</span>
                {:else}
                    <span class="text-sm text-gray-400">
                        Captioning… {captionProcessed}/{captionTotal} ({captionPct}%)
                        {#if estimatedRemainingSeconds != null}
                            · ~{formatEta(estimatedRemainingSeconds)} remaining
                        {/if}
                    </span>
                {/if}
            {:else if currentDataset}
                <p class="text-sm text-gray-400">
                    {currentDataset.image_count} images · {currentDataset.has_caption} captioned ·
                    {currentDataset.has_toml}
                    with TOML
                </p>
            {:else}
                <p class="text-sm text-gray-400">...</p>
            {/if}

            {#if isBatchCaptioning && !isStopping}
                <div class="absolute right-0 -bottom-1 left-0 h-0.5 bg-bg">
                    <div
                        class="h-full rounded-full bg-accent transition-all duration-300 ease-out"
                        style:width="{captionPct}%"
                    ></div>
                </div>
            {/if}
        </div>
    </div>
    {#if canUpload}
        <button
            class="shrink-0 cursor-pointer rounded-lg p-2 text-gray-400 transition-colors hover:bg-white/10 hover:text-white"
            title="Add files to this dataset"
            onclick={onupload}
        >
            <SvgUpload class="h-5 w-5" />
        </button>
    {/if}
    <button
        class="shrink-0 cursor-pointer rounded-lg p-2 text-gray-400 transition-colors hover:bg-white/10 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
        title="Refresh dataset from disk"
        disabled={isRefreshing}
        onclick={onrefresh}
    >
        {#if isRefreshing}
            <SvgSpinner class="h-5 w-5 animate-spin" />
        {:else}
            <SvgRefresh class="h-5 w-5" />
        {/if}
    </button>
</Topbar>
