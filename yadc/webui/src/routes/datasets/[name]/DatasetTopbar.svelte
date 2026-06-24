<script lang="ts">
    import type { DatasetInfo } from '$lib/stores/dataset';
    import { captionTimingRing, captioningStatuses } from '$lib/stores/caption';
    import { taggingStatuses } from '$lib/stores/tagging';
    import { formatEta } from '$lib/format';
    import { computeEtaSeconds, effectiveConcurrency } from '$lib/eta';
    import Topbar from '$lib/components/ui/Topbar.svelte';
    import SvgChevronLeft from '$lib/icons/SvgChevronLeft.svelte';
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import SvgUpload from '$lib/icons/SvgUpload.svelte';

    interface Props {
        /** The current dataset name (shown as the page title; also keys the
         *  per-dataset captioning/tagging status stores). */
        datasetName: string;
        /** The current dataset info (for the source badge and stats). */
        currentDataset: DatasetInfo | null;
        /** True when a manual refresh is in progress. */
        isRefreshing: boolean;
        /** True when uploads are allowed (managed dataset + no batch captioning). */
        canUpload: boolean;
        /** Called when the user clicks the refresh button. */
        onrefresh: () => void;
        /** Called when the user clicks the upload button. */
        onupload: () => void;
    }

    let { datasetName, currentDataset, isRefreshing, canUpload, onrefresh, onupload }: Props =
        $props();

    // --- Job status (subscribed directly — this topbar is route-scoped and
    // only renders for the open dataset). ---

    let captionStatus = $derived($captioningStatuses.get(datasetName));
    let tagStatus = $derived($taggingStatuses.get(datasetName));

    let isBatchCaptioning = $derived(
        captionStatus?.status === 'starting' ||
            captionStatus?.status === 'running' ||
            captionStatus?.status === 'stopping'
    );
    let isBatchTagging = $derived(
        tagStatus?.status === 'running' || tagStatus?.status === 'stopping'
    );

    function _pct(processed: number, total: number): number {
        return total > 0 ? Math.round((processed / total) * 100) : 0;
    }

    let captionPct = $derived(_pct(captionStatus?.processed ?? 0, captionStatus?.total ?? 0));
    let tagPct = $derived(_pct(tagStatus?.processed ?? 0, tagStatus?.total ?? 0));

    function _ringForStatus(
        ring: Record<string, number[]>,
        apiUrl?: string,
        model?: string
    ): number[] {
        if (!apiUrl || !model) {
            return [];
        }
        return ring[`${apiUrl}#${model}`] ?? [];
    }

    /** Formats the running-state detail suffix for captioning (concurrency,
     *  elapsed, ETA) as a pre-computed string, keeping the progress-row
     *  snippet free of domain logic. */
    function _captionDetail(status: NonNullable<typeof captionStatus>): string {
        if (status.status !== 'running') {
            return '';
        }
        const ring = _ringForStatus($captionTimingRing, status.api_url, status.api_model_name);
        const concurrent = effectiveConcurrency(
            status.max_concurrent,
            status.total,
            status.processed
        );
        const eta = computeEtaSeconds({
            total: status.total,
            processed: status.processed,
            ring,
            maxConcurrent: concurrent
        });
        const parts: string[] = [];
        if (concurrent > 1) {
            parts.push(`${concurrent} concurrent`);
        }
        if ((status.elapsed ?? 0) > 0) {
            parts.push(`${formatEta(Math.round(status.elapsed))} elapsed`);
        }
        if (eta != null) {
            parts.push(`~${formatEta(eta)} remaining`);
        }
        return parts.length ? `· ${parts.join(' · ')}` : '';
    }

    /** Tagging is single-subprocess serial, so ETA is trivial: per-image
     *  wall time = elapsed / processed, applied to the remaining count. No
     *  timing ring needed (TagJobInfo carries no concurrency field). */
    function _tagDetail(status: NonNullable<typeof tagStatus>): string {
        if (status.status !== 'running') {
            return '';
        }
        const parts: string[] = [];
        if ((status.elapsed ?? 0) > 0) {
            parts.push(`${formatEta(Math.round(status.elapsed))} elapsed`);
        }
        if (status.processed > 0 && status.total > status.processed) {
            const perImage = status.elapsed / status.processed;
            const remaining = (status.total - status.processed) * perImage;
            parts.push(`~${formatEta(Math.round(remaining))} remaining`);
        }
        return parts.length ? `· ${parts.join(' · ')}` : '';
    }

    let captionDetail = $derived(captionStatus ? _captionDetail(captionStatus) : '');
    let tagDetail = $derived(tagStatus ? _tagDetail(tagStatus) : '');

    // Map the per-job status enums onto the narrower set the progress row
    // renders (it only distinguishes starting / running / stopping).
    let captionProgressStatus = $derived(
        !captionStatus || captionStatus.status === 'done' || captionStatus.status === 'error'
            ? 'running'
            : (captionStatus.status as 'starting' | 'running' | 'stopping')
    );
    let tagProgressStatus = $derived(
        !tagStatus || tagStatus.status === 'done' || tagStatus.status === 'error'
            ? 'running'
            : (tagStatus.status as 'starting' | 'running' | 'stopping')
    );

    // Bundle each active job into a single render descriptor so the markup
    // below is one loop, not two near-identical blocks. Null when idle.
    interface ProgressView {
        verb: string;
        status: 'starting' | 'running' | 'stopping';
        processed: number;
        total: number;
        pct: number;
        detail: string;
    }

    let captionProgress = $derived<ProgressView | null>(
        isBatchCaptioning
            ? {
                  verb: 'Captioning',
                  status: captionProgressStatus,
                  processed: captionStatus?.processed ?? 0,
                  total: captionStatus?.total ?? 0,
                  pct: captionPct,
                  detail: captionDetail
              }
            : null
    );
    let tagProgress = $derived<ProgressView | null>(
        isBatchTagging
            ? {
                  verb: 'Tagging',
                  status: tagProgressStatus,
                  processed: tagStatus?.processed ?? 0,
                  total: tagStatus?.total ?? 0,
                  pct: tagPct,
                  detail: tagDetail
              }
            : null
    );

    /** Captioning first, then tagging — stable display order when both run. */
    let activeJobs = $derived(
        [captionProgress, tagProgress].filter((p): p is ProgressView => p !== null)
    );

    // One spinner if either job is active; yellow if either is stopping.
    let anyJobActive = $derived(isBatchCaptioning || isBatchTagging);
    let anyStopping = $derived(
        captionStatus?.status === 'stopping' || tagStatus?.status === 'stopping'
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
            {#if anyJobActive}
                <SvgSpinner
                    class="h-4 w-4 shrink-0 animate-spin {anyStopping
                        ? 'text-yellow-400'
                        : 'text-accent'}"
                />
            {/if}
        </div>

        {#snippet progressRow(job: ProgressView)}
            <div class="relative mt-0.5 flex items-center">
                {#if job.status === 'stopping'}
                    <span class="text-sm text-yellow-300">{job.verb}… (stopping)</span>
                {:else if job.status === 'starting'}
                    <span class="text-sm text-gray-400">{job.verb}… (starting)</span>
                {:else}
                    <span class="text-sm text-gray-400">
                        {job.verb}… {job.processed}/{job.total} ({job.pct}%){job.detail
                            ? ` ${job.detail}`
                            : ''}
                    </span>
                {/if}

                {#if job.status !== 'stopping'}
                    <div class="absolute right-0 -bottom-1 left-0 h-0.5 bg-bg">
                        <div
                            class="h-full rounded-full bg-accent transition-all duration-300 ease-out"
                            style:width="{job.pct}%"
                        ></div>
                    </div>
                {/if}
            </div>
        {/snippet}

        {#if activeJobs.length > 0}
            {#each activeJobs as job (job.verb)}
                {@render progressRow(job)}
            {/each}
        {:else if currentDataset}
            <p class="mt-0.5 text-sm text-gray-400">
                {currentDataset.image_count} images · {currentDataset.has_caption} captioned ·
                {currentDataset.has_toml}
                with TOML
            </p>
        {:else}
            <p class="mt-0.5 text-sm text-gray-400">...</p>
        {/if}
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
