<script lang="ts">
  import { onMount } from "svelte";
  import { browser } from "$app/environment";
  import { API_BASE } from "$lib/api";
  import { TypedEventSource } from "$lib/events";
  import { CaptioningStatusZ, type CaptioningStatus } from "$lib/stores/captioning";
  import SvgSpinner from "$lib/icons/SvgSpinner.svelte";

  interface Props {
    /** The dataset being captioned. */
    datasetName: string;
    /** Called when the job finishes (done, error, or idle). */
    ondone?: () => void;
  }

  let { datasetName, ondone }: Props = $props();

  // --- State ---

  let status: CaptioningStatus = $state({
    status: "idle",
    dataset_name: datasetName,
    processed: 0,
    total: 0,
    errors: 0,
    error: null,
  });

  let isStopping = $state(false);
  let isConnecting = $state(true);
  let connectionError: string | null = $state(null);

  // --- Derived ---

  let pct = $derived.by(() => {
    if (status.total <= 0) return 0;
    return Math.round((status.processed / status.total) * 100);
  });

  let progressLabel = $derived.by(() => {
    if (status.total > 0) return `${status.processed} / ${status.total}`;
    return "Starting…";
  });

  let isTerminal = $derived(
    status.status === "done" || status.status === "error" || status.status === "idle",
  );

  // --- SSE subscription ---

  let es: TypedEventSource | null = $state(null);

  $effect(() => {
    const name = datasetName;
    if (!browser || !name) return;

    // Clean up previous connection
    es?.close();
    es = null;
    isConnecting = true;
    connectionError = null;

    const url = `${API_BASE}/api/datasets/${encodeURIComponent(name)}/caption/status`;
    const source = new TypedEventSource(url);
    es = source;

    source.listen(
      "captioning_status",
      CaptioningStatusZ,
      (event) => {
        status = event;
        isConnecting = false;

        if (event.status === "done" || event.status === "error") {
          // Close after a short delay so the user sees the final state
          setTimeout(() => {
            source.close();
            ondone?.();
          }, 2000);
        }
      },
      (err) => {
        console.error("[CaptionProgress] SSE error:", err);
        connectionError = err.message;
        isConnecting = false;
      },
    );

    source.onerror = () => {
      // EventSource natively fires this on connection failure
      isConnecting = false;
      if (source.readyState === EventSource.CLOSED) {
        connectionError = "Connection lost";
      }
    };

    return () => {
      source.close();
    };
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
      {#if isConnecting}
        <SvgSpinner class="h-4 w-4 animate-spin text-accent" />
        <span class="text-sm font-medium text-gray-300">Connecting…</span>
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
  {#if status.total > 0 || isConnecting}
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

  <!-- Connection error -->
  {#if connectionError && !isConnecting}
    <p class="text-xs text-error">{connectionError}</p>
  {/if}
</div>
