<script lang="ts">
  import type { Snippet } from "svelte";
  import { page } from "$app/stores";
  import "./layout.css";
  import SettingsDialog from "$lib/components/dialogs/SettingsDialog.svelte";
  import ExportDialog from "$lib/components/dialogs/ExportDialog.svelte";
  import SvgUpload from "$lib/icons/SvgUpload.svelte";
  import SvgSettings from "$lib/icons/SvgSettings.svelte";
  import type { ExportResult } from "$lib/stores/configs";

  interface Props {
    children: Snippet;
  }

  let { children }: Props = $props();

  let currentHash = $derived($page.url.hash);
  let isDatasetPage = $derived(currentHash === "#/" || currentHash.startsWith("#/datasets/"));
  let isTemplatesPage = $derived(currentHash.startsWith("#/templates"));
  let datasetName = $derived(isDatasetPage ? $page.params.name : null);

  let showSettings = $state(false);
  let showExport = $state(false);

  let exportResult: ExportResult | null = $state(null);

  $effect(() => {
    document.getElementById("yadc-loading-screen")?.remove();
  });

  function handleExported(result: ExportResult) {
    exportResult = result;
    // Auto-clear after 5 seconds
    setTimeout(() => (exportResult = null), 5000);
  }
</script>

<div class="app-shell">
  <nav class="app-nav">
    <div class="nav-brand">
      <a href="#/">yadc</a>
    </div>
    <div class="nav-links">
      <a href="#/" class:active={isDatasetPage}>Datasets</a>
      <a href="#/templates" class:active={isTemplatesPage}>Templates</a>
      {#if isDatasetPage && datasetName}
        <span class="nav-separator">/</span>
        <span class="nav-current">{datasetName}</span>
      {/if}
    </div>
    <div class="nav-actions">
      <button class="nav-btn" onclick={() => (showExport = true)} title="Export">
        <SvgUpload class="w-6 h-6" />
      </button>
      <button class="nav-btn" onclick={() => (showSettings = true)} title="Settings">
        <SvgSettings class="w-6 h-6" />
      </button>
    </div>
  </nav>

  <main class="app-main">
    {#if exportResult}
      <div class="export-toast">
        Exported {exportResult.count} images → {exportResult.output}
      </div>
    {/if}
    {@render children()}
  </main>
</div>

<SettingsDialog open={showSettings} onclose={() => (showSettings = false)} />
<ExportDialog open={showExport} onclose={() => (showExport = false)} onexported={handleExported} />

<style>
  .app-shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  .app-nav {
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 0.75rem 1.5rem;
    background: var(--color-surface);
    border-bottom: 1px solid var(--color-border);
  }

  .nav-brand a {
    font-weight: 700;
    font-size: 1.2rem;
    color: var(--color-accent);
    text-decoration: none;
  }

  .nav-links {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    flex: 1;
  }

  .nav-links a {
    color: var(--color-muted);
    text-decoration: none;
    transition: color 0.2s;
  }

  .nav-links a:hover {
    color: var(--color-fg);
  }

  .nav-links a.active {
    color: var(--color-fg);
  }

  .nav-separator {
    color: var(--color-muted);
    opacity: 0.4;
  }

  .nav-current {
    color: var(--color-fg);
    font-size: 0.875rem;
  }

  .nav-actions {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .nav-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 0.375rem;
    color: var(--color-muted);
    background: transparent;
    border: none;
    cursor: pointer;
    transition: color 0.2s, background-color 0.2s;
  }

  .nav-btn:hover {
    color: var(--color-fg);
    background: var(--color-bg);
  }

  .app-main {
    flex: 1;
    padding: 1.5rem;
    min-height: 0;
    overflow-y: auto;
  }

  .export-toast {
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    padding: 0.75rem 1.25rem;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 0.5rem;
    color: var(--color-fg);
    font-size: 0.875rem;
    z-index: 40;
    animation: toast-in 0.2s ease-out;
  }

  @keyframes toast-in {
    from {
      opacity: 0;
      transform: translateY(0.5rem);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>
