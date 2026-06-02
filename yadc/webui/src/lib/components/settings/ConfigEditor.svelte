<script lang="ts">
  import CodeMirror from "$lib/components/ui/CodeMirror.svelte";
  import ConfirmDelete from "$lib/components/ui/ConfirmDelete.svelte";
  import SpinnerBlock from "$lib/components/ui/SpinnerBlock.svelte";
  import SvgDelete from "$lib/icons/SvgDelete.svelte";
  import { minimalSetup } from "codemirror";
  import { EditorView } from "@codemirror/view";
  import { StreamLanguage } from "@codemirror/language";
  import { toml } from "@codemirror/legacy-modes/mode/toml";
  import { fetchConfigs, fetchConfig, updateConfig, deleteConfig, type DatasetConfig } from "$lib/stores/configs";

  interface Props {
    /** Reload data when this becomes true (e.g. dialog opens). */
    open: boolean;
  }

  let { open }: Props = $props();

  let configList: DatasetConfig[] = $state([]);
  let selectedConfigName = $state("");
  let configContent = $state("");
  let isLoadingConfig = $state(false);
  let configDirty = $state(false);
  let isSavingConfig = $state(false);
  let configError: string | null = $state(null);
  let confirmDeleteConfig: string | null = $state(null);

  const tomlExtensions = [
    minimalSetup,
    StreamLanguage.define(toml),
    EditorView.lineWrapping,
  ];

  $effect(() => {
    if (open) {
      configError = null;
      configDirty = false;
      confirmDeleteConfig = null;
      loadConfigs();
    }
  });

  async function loadConfigs() {
    try {
      configList = await fetchConfigs();
    } catch (e) {
      configError = e instanceof Error ? e.message : "Failed to load configs";
    }
  }

  async function selectConfig(name: string) {
    selectedConfigName = name;
    isLoadingConfig = true;
    configError = null;
    configDirty = false;
    confirmDeleteConfig = null;
    try {
      const detail = await fetchConfig(name);
      configContent = detail.content;
    } catch (e) {
      configError = e instanceof Error ? e.message : "Failed to load config";
      configContent = "";
    } finally {
      isLoadingConfig = false;
    }
  }

  function handleConfigContentChange(value: string) {
    configContent = value;
    configDirty = true;
  }

  async function saveConfigContent() {
    if (!selectedConfigName) return;
    isSavingConfig = true;
    configError = null;
    try {
      const detail = await updateConfig(selectedConfigName, configContent);
      configContent = detail.content;
      configDirty = false;
      await loadConfigs();
    } catch (e) {
      configError = e instanceof Error ? e.message : "Failed to save config";
    } finally {
      isSavingConfig = false;
    }
  }

  async function handleDeleteConfig(name: string) {
    try {
      await deleteConfig(name);
      confirmDeleteConfig = null;
      if (selectedConfigName === name) {
        selectedConfigName = "";
        configContent = "";
      }
      await loadConfigs();
    } catch (e) {
      configError = e instanceof Error ? e.message : "Failed to delete config";
    }
  }
</script>

{#if configError}
  <div class="alert-error">{configError}</div>
{/if}

<div class="flex gap-4" style="height: 50vh;">
  <!-- Config list sidebar -->
  <div class="w-48 shrink-0 space-y-1 overflow-y-auto">
    {#each configList as cfg (cfg.name)}
      <div
        role="button"
        tabindex="0"
        class="w-full text-left px-3 py-2 text-sm rounded-lg transition-colors cursor-pointer group {selectedConfigName === cfg.name ? 'bg-bg text-accent border border-border' : 'text-gray-300 hover:bg-bg/50'}"
        onclick={() => selectConfig(cfg.name)}
        onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') selectConfig(cfg.name); }}
      >
        <div class="flex items-center justify-between gap-1">
          <span class="truncate">{cfg.name}</span>
          <button
            class="p-0.5 text-gray-500 hover:text-error opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer shrink-0"
            title="Delete config"
            onclick={(e) => { e.stopPropagation(); confirmDeleteConfig = cfg.name; }}
          >
            <SvgDelete class="h-3 w-3" />
          </button>
        </div>
      </div>
    {/each}
    {#if configList.length === 0}
      <p class="text-gray-500 text-xs text-center py-4">No datasets</p>
    {/if}
  </div>

  <!-- Config editor -->
  <div class="flex-1 flex flex-col min-w-0">
    {#if isLoadingConfig}
      <SpinnerBlock class="py-12" />
    {:else if selectedConfigName}
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-sm font-medium text-gray-300">{selectedConfigName}.toml</h3>
        <div class="flex gap-2">
          {#if configDirty}
            <button
              class="btn-primary px-3 py-1.5 text-xs"
              onclick={saveConfigContent}
              disabled={isSavingConfig}
            >
              {isSavingConfig ? "Saving…" : "Save"}
            </button>
          {/if}
        </div>
      </div>
      <div class="flex-1 min-h-0 overflow-hidden">
        <CodeMirror
          doc={configContent}
          extensions={tomlExtensions}
          onchange={handleConfigContentChange}
          class="h-full"
        />
      </div>
    {:else}
      <div class="flex items-center justify-center h-full text-gray-500 text-sm">
        Select a config to edit
      </div>
    {/if}
  </div>
</div>

<ConfirmDelete
  open={confirmDeleteConfig !== null}
  oncancel={() => (confirmDeleteConfig = null)}
  onconfirm={() => handleDeleteConfig(confirmDeleteConfig!)}
>
  Delete config for <strong>{confirmDeleteConfig}</strong>? This will unregister the dataset.
</ConfirmDelete>
