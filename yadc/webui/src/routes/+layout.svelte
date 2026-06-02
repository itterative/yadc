<script lang="ts">
  import type { Snippet } from "svelte";
  import { page } from "$app/stores";
  import "./layout.css";
  import SettingsDialog from "$lib/components/SettingsDialog.svelte";
  import ExportDialog from "$lib/components/ExportDialog.svelte";
  import type { ExportResult } from "$lib/stores/configs";

  interface Props {
    children: Snippet;
  }

  let { children }: Props = $props();

  let currentPath = $derived($page.url.pathname);
  let isDatasetPage = $derived(currentPath.startsWith("/datasets/"));
  let datasetName = $derived(isDatasetPage ? $page.params.name : null);

  let showSettings = $state(false);
  let showExport = $state(false);

  let exportResult: ExportResult | null = $state(null);

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
      <a href="#/" class:active={!isDatasetPage}>Datasets</a>
      {#if isDatasetPage && datasetName}
        <span class="nav-separator">/</span>
        <span class="nav-current">{datasetName}</span>
      {/if}
    </div>
    <div class="nav-actions">
      <button class="nav-btn" onclick={() => (showExport = true)} title="Export">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      </button>
      <button class="nav-btn" onclick={() => (showSettings = true)} title="Settings">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
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
    gap: 0.5rem;
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
    gap: 0.25rem;
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
