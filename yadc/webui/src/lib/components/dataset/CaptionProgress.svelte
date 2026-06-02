<script lang="ts">
  import { captioningStatus, type CaptioningStatus } from "$lib/stores/events";
  import SvgSpinner from "$lib/icons/SvgSpinner.svelte";
  import { API_BASE } from "$lib/api";

  interface Props {
    /** The dataset being captioned. */
    datasetName: string;
    /** Called when the job finishes (done, error, or idle). */
    ondone?: () => void;
  }

  let { datasetName, ondone }: Props = $props();

  // --- Derived ---

  // Pick the latest status for *this* dataset from the global SSE store
  let status: CaptioningStatus = $derived(
    $captioningStatus?.dataset_name === datasetName
      ? $captioningStatus
      : { status: "idle" as const, dataset_name: "", processed: 0, total: 0, errors: 0, job_id: "", error: null },
  );

  let isStopping = $state(false);

  // Don't react to terminal states until we've seen the job go active (running/stopping).
  // This prevents stale "done" events from previous runs from immediately dismissing the component.
  let seenActive = $state(false);

  $effect(() => {
    if (status.status === "running" || status.status === "stopping") {
      seenActive = true;
    }
  });

  let pct = $derived.by(() => {
    if (status.total <= 0) return 0;
    return Math.round((status.processed / status.total) * 100);
  });

  let progressLabel = $derived.by(() => {
    if (status.total > 0) return `${status.processed} / ${status.total}`;
    return "Starting…";
  });

  let isTerminal = $derived(seenActive && (status.status === "done" || status.status === "error"));

  // Fire ondone when the job reaches a terminal state
  let prevTerminal = $state(false);
  $effect(() => {
    if (isTerminal && !prevTerminal) {
      prevTerminal = true;
      setTimeout(() => ondone?.(), 2000);
    }
  });

  // --- Stop captioning ---

  async function handleStop() {
    isStopping = true;
    try {
      const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`,
        { method: "DELETE" },
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        console.error("[CaptionProgress] stop failed:", body.error || res.status);
      }
    } catch (e) {
      console.error("[CaptionProgress] stop request failed:", e);
    } finally {
      isStopping = false;
    }
  }
</script>

<div class="rounded-xl bg-surface border border-border p-4 space-y-3">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div class="flex items-center gap-2">
      {#if !seenActive}
        <SvgSpinner class="h-4 w-4 animate-spin text-accent" />
        <span class="text-sm font-medium text-gray-300">Starting…</span>
      {:else if status.status === "running"}
        <SvgSpinner class="h-4 w-4 animate-spin text-accent" />
        <span class="text-sm font-medium text-gray-300">
          Captioning: {datasetName}
        </span>
      {:else if status.status === "stopping"}
        <SvgSpinner class="h-4 w-4 animate-spin text-yellow-400" />
        <span class="text-sm font-medium text-yellow-300">Stopping…</span>
      {:else if status.status === "done"}
        <span class="text-success text-lg">✓</span>
        <span class="text-sm font-medium text-success">Complete</span>
      {:else if status.status === "error"}
        <span class="text-error text-lg">✗</span>
        <span class="text-sm font-medium text-error">Error</span>
      {:else}
        <span class="text-sm text-gray-400">Idle</span>
      {/if}
    </div>

    {#if !isTerminal}
      <button
        class="px-3 py-1.5 text-xs rounded-lg bg-error/20 border border-error/30 text-error hover:bg-error/30 transition-colors cursor-pointer disabled:opacity-50"
        onclick={handleStop}
        disabled={isStopping}
      >
        {isStopping ? "Stopping…" : "Stop"}
      </button>
    {/if}
  </div>

  <!-- Progress bar -->
  {#if status.total > 0}
    <div class="space-y-1">
      <div class="flex items-center justify-between text-xs text-gray-400">
        <span>{progressLabel}</span>
        <span>{pct}%</span>
      </div>
      <div class="h-2 rounded-full bg-bg overflow-hidden">
        <div
          class="h-full rounded-full transition-all duration-300 ease-out"
          class:bg-accent={status.status === "running"}
          class:bg-yellow-400={status.status === "stopping"}
          class:bg-success={status.status === "done"}
          class:bg-error={status.status === "error"}
          style:width={`${pct}%`}
        ></div>
      </div>
    </div>
  {/if}

  <!-- Stats -->
  {#if status.processed > 0 || status.errors > 0}
    <div class="flex items-center gap-4 text-xs text-gray-400">
      {#if status.processed > 0}
        <span>Processed: {status.processed}</span>
      {/if}
      {#if status.errors > 0}
        <span class="text-error">Errors: {status.errors}</span>
      {/if}
    </div>
  {/if}

  <!-- Error detail -->
  {#if status.error}
    <p class="text-xs text-error bg-error/10 rounded-lg p-2">{status.error}</p>
  {/if}
</div>
