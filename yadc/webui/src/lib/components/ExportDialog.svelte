<script lang="ts">
  import Dialog from "$lib/components/Dialog.svelte";
  import CodeMirror from "$lib/components/CodeMirror.svelte";
  import Checkbox from "$lib/components/Checkbox.svelte";
  import SvgClose from "$lib/icons/SvgClose.svelte";
  import SvgSpinner from "$lib/icons/SvgSpinner.svelte";
  import {
    fetchExportBackends,
    runExport,
    type ExportBackend,
    type ExportResult,
  } from "$lib/stores/configs";
  import { fetchDatasets, type DatasetInfo } from "$lib/stores/datasetImages";

  interface Props {
    open: boolean;
    onclose: () => void;
    /** Pre-selected dataset name. */
    datasetName?: string;
    /** Called after a successful export. */
    onexported?: (result: ExportResult) => void;
  }

  let { open, onclose, datasetName = "", onexported }: Props = $props();

  // --- Data ---
  let datasets: DatasetInfo[] = $state([]);
  let backends: ExportBackend[] = $state([]);
  let isLoading = $state(false);
  let isExporting = $state(false);
  let error: string | null = $state(null);

  // --- Form state ---
  let selectedDataset = $state(datasetName);
  let selectedBackend = $state("sd-scripts");
  let selectedFormat = $state("jsonl");
  let source: "caption" | "draft" = $state("caption");
  let draftName = $state("");
  let outputPath = $state("");
  let append = $state(false);
  let captionExtension = $state(".txt");

  // --- Result ---
  let result: ExportResult | null = $state(null);

  // --- Derived ---
  let currentBackend = $derived(backends.find((b) => b.name === selectedBackend));
  let formats = $derived(currentBackend?.formats ?? ["jsonl"]);

  // --- Load data on open ---
  $effect(() => {
    if (open) {
      result = null;
      error = null;
      selectedDataset = datasetName;
      load();
    }
  });

  // When backend changes, reset format to first available
  $effect(() => {
    if (currentBackend && !currentBackend.formats.includes(selectedFormat)) {
      selectedFormat = currentBackend.formats[0];
    }
  });

  async function load() {
    isLoading = true;
    error = null;
    try {
      const [ds, be] = await Promise.all([fetchDatasets(), fetchExportBackends()]);
      datasets = ds;
      backends = be;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load export options";
    } finally {
      isLoading = false;
    }
  }

  async function handleExport() {
    if (!selectedDataset) {
      error = "Please select a dataset";
      return;
    }

    isExporting = true;
    error = null;
    result = null;

    try {
      const opts: Record<string, unknown> = {
        dataset: selectedDataset,
        backend: selectedBackend,
        format: selectedFormat,
        source,
        append,
        caption_extension: captionExtension,
      };

      if (source === "draft" && draftName.trim()) {
        opts.draft = draftName.trim();
      }
      if (outputPath.trim()) {
        opts.output = outputPath.trim();
      }

      const res = await runExport(opts as Parameters<typeof runExport>[0]);
      result = res;
      onexported?.(res);
    } catch (e) {
      error = e instanceof Error ? e.message : "Export failed";
    } finally {
      isExporting = false;
    }
  }
</script>

<Dialog
  class="w-full max-w-lg max-h-[85vh] overflow-y-auto bg-surface rounded-xl shadow-2xl border border-border m-auto"
  {open}
  onclose={onclose}
>
  <div class="p-6 space-y-5">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <h2 class="text-lg font-semibold text-white">Export Dataset</h2>
      <button
        class="p-1 text-gray-400 hover:text-white transition-colors cursor-pointer"
        onclick={onclose}
      >
        <SvgClose class="h-5 w-5" />
      </button>
    </div>

    {#if error}
      <div class="p-3 rounded-lg bg-error/10 border border-error/20 text-error text-sm">{error}</div>
    {/if}

    {#if result}
      <!-- Success state -->
      <div class="p-4 rounded-lg bg-green-900/20 border border-green-700/30 space-y-2">
        <p class="text-green-300 font-medium text-sm">Export complete</p>
        <div class="text-sm text-gray-300 space-y-1">
          <p><span class="text-gray-500">Images exported:</span> {result.count}</p>
          <p><span class="text-gray-500">Format:</span> {result.format}</p>
          <p><span class="text-gray-500">Output:</span> {result.output}</p>
        </div>
        <button
          class="mt-2 px-4 py-2 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 cursor-pointer"
          onclick={() => (result = null)}
        >
          Export Again
        </button>
      </div>
    {:else if isLoading}
      <div class="flex items-center justify-center py-8">
        <SvgSpinner class="h-6 w-6 animate-spin text-gray-400" />
      </div>
    {:else}
      <!-- Dataset -->
      <div>
        <label class="block text-sm text-gray-400 mb-1" for="export-dataset">Dataset</label>
        <select
          id="export-dataset"
          class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none cursor-pointer"
          bind:value={selectedDataset}
        >
          <option value="" disabled>Select dataset</option>
          {#each datasets as ds (ds.name)}
            <option value={ds.name}>{ds.name} ({ds.image_count} images)</option>
          {/each}
        </select>
      </div>

      <!-- Backend & Format -->
      <div class="grid grid-cols-2 gap-3">
        <div>
          <label class="block text-sm text-gray-400 mb-1" for="export-backend">Backend</label>
          <select
            id="export-backend"
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none cursor-pointer"
            bind:value={selectedBackend}
          >
            {#each backends as b (b.name)}
              <option value={b.name}>{b.name}</option>
            {/each}
          </select>
          {#if currentBackend}
            <p class="text-xs text-gray-500 mt-1">{currentBackend.description}</p>
          {/if}
        </div>
        <div>
          <label class="block text-sm text-gray-400 mb-1" for="export-format">Format</label>
          <select
            id="export-format"
            class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none cursor-pointer"
            bind:value={selectedFormat}
          >
            {#each formats as f (f)}
              <option value={f}>{f}</option>
            {/each}
          </select>
        </div>
      </div>

      <!-- Source -->
      <div>
        <label class="block text-sm text-gray-400 mb-1">Source</label>
        <div class="flex gap-4">
          <label class="flex items-center gap-2 text-sm text-gray-200 cursor-pointer">
            <input type="radio" name="export-source" value="caption" bind:group={source} />
            Captions
          </label>
          <label class="flex items-center gap-2 text-sm text-gray-200 cursor-pointer">
            <input type="radio" name="export-source" value="draft" bind:group={source} />
            Draft
          </label>
        </div>
        {#if source === "draft"}
          <input
            type="text"
            bind:value={draftName}
            class="mt-2 w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
            placeholder="Draft name"
          />
        {/if}
      </div>

      <!-- Output path -->
      <div>
        <label class="block text-sm text-gray-400 mb-1" for="export-output">
          Output Path
          <span class="text-gray-600 ml-1">(optional — auto-detected if empty)</span>
        </label>
        <input
          id="export-output"
          type="text"
          bind:value={outputPath}
          class="w-full rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
          placeholder="e.g. /data/training/metadata.jsonl"
        />
      </div>

      <!-- Options -->
      <div class="flex items-center gap-6">
        <div class="flex items-center gap-2">
          <Checkbox id="export-append" bind:checked={append} />
          <label class="text-sm text-gray-300 cursor-pointer" for="export-append">Append</label>
        </div>
        <div class="flex items-center gap-2">
          <label class="text-sm text-gray-400" for="export-ext">Extension</label>
          <input
            id="export-ext"
            type="text"
            bind:value={captionExtension}
            class="w-20 rounded-lg bg-bg border border-border px-2 py-1 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
          />
        </div>
      </div>

      <!-- Footer -->
      <div class="flex gap-2 justify-end pt-2 border-t border-border">
        <button
          class="px-4 py-2 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 cursor-pointer"
          onclick={onclose}
        >
          Cancel
        </button>
        <button
          class="px-4 py-2 text-sm rounded-lg bg-accent hover:bg-accent-hover text-black font-medium cursor-pointer disabled:opacity-50"
          onclick={handleExport}
          disabled={isExporting || !selectedDataset}
        >
          {isExporting ? "Exporting…" : "Export"}
        </button>
      </div>
    {/if}
  </div>
</Dialog>
