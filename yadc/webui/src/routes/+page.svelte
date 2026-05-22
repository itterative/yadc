<script lang="ts">
  import { onMount } from "svelte";
  import { browser } from "$app/environment";
  import { TypedEventSource } from "$lib/events";
  import {
    captioningStatus,
    CaptioningStatusZ,
    PingEventZ,
  } from "$lib/stores/captioning";
  import { API_BASE } from "$lib/api";

  interface Dataset {
    name: string;
    path: string;
    image_count?: number;
  }

  let datasets: Dataset[] = $state([]);
  let loading = $state(true);
  let error = $state("");

  let eventSource: TypedEventSource | null = null;

  onMount(() => {
    if (!browser) {
      return;
    }

    // Connect SSE event stream with auto-reconnect
    function connect() {
      if (eventSource !== null && eventSource.readyState !== EventSource.CLOSED) {
        return;
      }

      try {
        eventSource = new TypedEventSource(`${API_BASE}/api/events`);
      } catch (e) {
        console.warn("Failed to connect to event stream", { error: e });
        return;
      }

      eventSource.listen("captioning_status", CaptioningStatusZ, (data) => {
        captioningStatus.set(data);
      });

      eventSource.listen("ping", PingEventZ, () => {
        // keepalive
      });

      eventSource.onerror = function () {
        eventSource?.close();
      };
    }

    connect();
    let reconnecting = false;

    const interval = window.setInterval(() => {
      if (eventSource === null) {
        connect();
      } else if (!reconnecting && eventSource.readyState === EventSource.CLOSED) {
        reconnecting = true;
        connect();
        reconnecting = false;
      }
    }, 5000);

    // Fetch datasets
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/datasets`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        datasets = await res.json();
      } catch (e) {
        error = e instanceof Error ? e.message : "Failed to load datasets";
      } finally {
        loading = false;
      }
    })();

    return () => {
      window.clearInterval(interval);
      eventSource?.close();
    };
  });
</script>

<div class="page-header">
  <h1>Datasets</h1>
</div>

{#if loading}
  <p class="text-dim">Loading datasets...</p>
{:else if error}
  <p class="text-error">Error: {error}</p>
{:else if datasets.length === 0}
  <div class="empty-state">
    <h2>No datasets found</h2>
    <p>Configure a dataset with a TOML file to get started.</p>
  </div>
{:else}
  <div class="dataset-grid">
    {#each datasets as dataset}
      <div class="dataset-card">
        <h3>{dataset.name}</h3>
        <p class="text-dim">{dataset.path}</p>
        {#if dataset.image_count !== undefined}
          <p class="text-dim">{dataset.image_count} images</p>
        {/if}
      </div>
    {/each}
  </div>
{/if}

<style>
  .page-header {
    margin-bottom: 1.5rem;
  }

  .page-header h1 {
    margin: 0;
    font-size: 1.5rem;
  }

  .text-dim {
    color: var(--color-text-dim);
  }

  .text-error {
    color: var(--color-error);
  }

  .empty-state {
    text-align: center;
    padding: 3rem;
    color: var(--color-text-dim);
  }

  .empty-state h2 {
    margin: 0 0 0.5rem;
  }

  .dataset-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
  }

  .dataset-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    transition: border-color 0.2s;
  }

  .dataset-card:hover {
    border-color: var(--color-accent);
  }

  .dataset-card h3 {
    margin: 0 0 0.25rem;
    color: var(--color-text);
  }
</style>
