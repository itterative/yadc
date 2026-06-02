<script lang="ts">
  import Dialog from "$lib/components/Dialog.svelte";
  import EnvManager from "$lib/components/EnvManager.svelte";
  import JinjaEditor from "$lib/components/JinjaEditor.svelte";
  import CodeMirror from "$lib/components/CodeMirror.svelte";
  import SvgClose from "$lib/icons/SvgClose.svelte";
  import SvgDelete from "$lib/icons/SvgDelete.svelte";
  import SvgPlus from "$lib/icons/SvgPlus.svelte";
  import SvgSpinner from "$lib/icons/SvgSpinner.svelte";
  import SvgFile from "$lib/icons/SvgFile.svelte";
  import { minimalSetup } from "codemirror";
  import { EditorView } from "@codemirror/view";
  import { StreamLanguage } from "@codemirror/language";
  import { toml } from "@codemirror/legacy-modes/mode/toml";
  import { fetchConfigs, fetchConfig, updateConfig, deleteConfig } from "$lib/stores/configs";
  import {
    fetchTemplates,
    fetchTemplate,
    saveTemplate,
    deleteTemplate,
    type TemplateListItem,
  } from "$lib/stores/templates";

  interface Props {
    open: boolean;
    onclose: () => void;
  }

  let { open, onclose }: Props = $props();

  // --- Tabs ---
  type Tab = "configs" | "templates" | "environments";
  let activeTab: Tab = $state("configs");

  // --- Config state ---
  let configList: { name: string; config_path: string }[] = $state([]);
  let selectedConfigName = $state("");
  let configContent = $state("");
  let isLoadingConfig = $state(false);
  let configDirty = $state(false);
  let isSavingConfig = $state(false);
  let configError: string | null = $state(null);
  let confirmDeleteConfig: string | null = $state(null);

  // --- Template state ---
  let templateList: TemplateListItem[] = $state([]);
  let selectedTemplateName = $state("");
  let templateContent = $state("");
  let templateSource: "user" | "builtin" | "" = $state("");
  let isLoadingTemplate = $state(false);
  let templateDirty = $state(false);
  let isSavingTemplate = $state(false);
  let templateError: string | null = $state(null);
  let confirmDeleteTemplate: string | null = $state(null);
  let isNewTemplate = $state(false);
  let newTemplateName = $state("");

  // --- Env state ---
  let showEnvManager = $state(false);

  // --- TOML editor extensions (static, like TomlEditor) ---
  const tomlExtensions = [
    minimalSetup,
    StreamLanguage.define(toml),
    EditorView.lineWrapping,
  ];

  // --- Load data on open ---
  $effect(() => {
    if (open) {
      configError = null;
      templateError = null;
      configDirty = false;
      templateDirty = false;
      confirmDeleteConfig = null;
      confirmDeleteTemplate = null;
      isNewTemplate = false;
      loadConfigs();
      loadTemplates();
    }
  });

  // ═══ Config section ═══

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

  // ═══ Template section ═══

  async function loadTemplates() {
    try {
      templateList = await fetchTemplates();
    } catch (e) {
      templateError = e instanceof Error ? e.message : "Failed to load templates";
    }
  }

  async function selectTemplate(name: string) {
    selectedTemplateName = name;
    isLoadingTemplate = true;
    templateError = null;
    templateDirty = false;
    confirmDeleteTemplate = null;
    isNewTemplate = false;
    try {
      const info = await fetchTemplate(name);
      templateContent = info.content;
      templateSource = info.source;
    } catch (e) {
      templateError = e instanceof Error ? e.message : "Failed to load template";
      templateContent = "";
      templateSource = "";
    } finally {
      isLoadingTemplate = false;
    }
  }

  function handleTemplateContentChange(value: string) {
    templateContent = value;
    templateDirty = true;
  }

  async function saveTemplateContent() {
    const name = isNewTemplate ? newTemplateName.trim() : selectedTemplateName;
    if (!name) {
      templateError = "Template name is required";
      return;
    }

    isSavingTemplate = true;
    templateError = null;
    try {
      const info = await saveTemplate(name, templateContent);
      isNewTemplate = false;
      selectedTemplateName = info.name;
      templateSource = info.source;
      templateDirty = false;
      await loadTemplates();
    } catch (e) {
      templateError = e instanceof Error ? e.message : "Failed to save template";
    } finally {
      isSavingTemplate = false;
    }
  }

  async function handleDeleteTemplate(name: string) {
    try {
      await deleteTemplate(name);
      confirmDeleteTemplate = null;
      if (selectedTemplateName === name) {
        selectedTemplateName = "";
        templateContent = "";
        templateSource = "";
      }
      await loadTemplates();
    } catch (e) {
      templateError = e instanceof Error ? e.message : "Failed to delete template";
    }
  }

  function startNewTemplate() {
    isNewTemplate = true;
    newTemplateName = "";
    templateContent = "";
    templateSource = "";
    templateDirty = false;
    templateError = null;
    selectedTemplateName = "";
  }

  function cancelNewTemplate() {
    isNewTemplate = false;
    newTemplateName = "";
    templateError = null;
  }

  // ═══ Tab styling ═══

  const tabClass = (active: boolean) =>
    `px-4 py-2 text-sm font-medium rounded-t-lg cursor-pointer transition-colors ${
      active ? "bg-surface text-accent border-b-2 border-accent" : "text-gray-400 hover:text-gray-200"
    }`;
</script>

<!-- Sub-dialogs -->
<EnvManager open={showEnvManager} onclose={() => (showEnvManager = false)} />

<Dialog
  class="w-full max-w-3xl max-h-[85vh] overflow-hidden bg-surface rounded-xl shadow-2xl border border-border m-auto flex flex-col"
  {open}
  onclose={onclose}
>
  <div class="flex flex-col h-full max-h-[85vh]">
    <!-- Header -->
    <div class="flex items-center justify-between p-5 pb-0">
      <h2 class="text-lg font-semibold text-white">Settings</h2>
      <button
        class="p-1 text-gray-400 hover:text-white transition-colors cursor-pointer"
        onclick={onclose}
      >
        <SvgClose class="h-5 w-5" />
      </button>
    </div>

    <!-- Tabs -->
    <div class="flex gap-1 px-5 pt-3 border-b border-border">
      <button class={tabClass(activeTab === "configs")} onclick={() => (activeTab = "configs")}>
        <span class="flex items-center gap-1.5">
          <SvgFile class="h-3.5 w-3.5" />
          Configs
        </span>
      </button>
      <button class={tabClass(activeTab === "templates")} onclick={() => (activeTab = "templates")}>
        Templates
      </button>
      <button class={tabClass(activeTab === "environments")} onclick={() => (activeTab = "environments")}>
        Environments
      </button>
    </div>

    <!-- Tab content -->
    <div class="flex-1 overflow-y-auto p-5 space-y-4">
      {#if configError && activeTab === "configs"}
        <div class="p-3 rounded-lg bg-error/10 border border-error/20 text-error text-sm">{configError}</div>
      {/if}
      {#if templateError && activeTab === "templates"}
        <div class="p-3 rounded-lg bg-error/10 border border-error/20 text-error text-sm">{templateError}</div>
      {/if}

      <!-- ═══ Configs tab ═══ -->
      {#if activeTab === "configs"}
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
              <div class="flex items-center justify-center py-12">
                <SvgSpinner class="h-6 w-6 animate-spin text-gray-400" />
              </div>
            {:else if selectedConfigName}
              <div class="flex items-center justify-between mb-2">
                <h3 class="text-sm font-medium text-gray-300">{selectedConfigName}.toml</h3>
                <div class="flex gap-2">
                  {#if configDirty}
                    <button
                      class="px-3 py-1.5 text-xs rounded-lg bg-accent hover:bg-accent-hover text-black font-medium cursor-pointer disabled:opacity-50"
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

        <!-- Delete config confirmation -->
        {#if confirmDeleteConfig}
          <div class="p-4 rounded-lg bg-error/10 border border-error/20">
            <p class="text-sm text-gray-200 mb-3">
              Delete config for <strong>{confirmDeleteConfig}</strong>? This will unregister the dataset.
            </p>
            <div class="flex gap-2 justify-end">
              <button
                class="px-3 py-1.5 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 cursor-pointer"
                onclick={() => (confirmDeleteConfig = null)}
              >
                Cancel
              </button>
              <button
                class="px-3 py-1.5 text-sm rounded-lg bg-error hover:bg-error/80 text-white cursor-pointer"
                onclick={() => handleDeleteConfig(confirmDeleteConfig!)}
              >
                Delete
              </button>
            </div>
          </div>
        {/if}

      <!-- ═══ Templates tab ═══ -->
      {:else if activeTab === "templates"}
        <div class="flex gap-4" style="height: 50vh;">
          <!-- Template list sidebar -->
          <div class="w-48 shrink-0 space-y-1 overflow-y-auto">
            {#each templateList as t (t.name)}
              <div
                role="button"
                tabindex="0"
                class="w-full text-left px-3 py-2 text-sm rounded-lg transition-colors cursor-pointer group {selectedTemplateName === t.name && !isNewTemplate ? 'bg-bg text-accent border border-border' : 'text-gray-300 hover:bg-bg/50'}"
                onclick={() => selectTemplate(t.name)}
                onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') selectTemplate(t.name); }}
              >
                <div class="flex items-center justify-between gap-1">
                  <span class="truncate">{t.name}</span>
                  <span class="flex items-center gap-0.5 shrink-0">
                    {#if t.source === "builtin"}
                      <span class="text-[10px] text-gray-500 uppercase">built-in</span>
                    {/if}
                    {#if t.source === "user"}
                      <button
                        class="p-0.5 text-gray-500 hover:text-error opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                        title="Delete template"
                        onclick={(e) => { e.stopPropagation(); confirmDeleteTemplate = t.name; }}
                      >
                        <SvgDelete class="h-3 w-3" />
                      </button>
                    {/if}
                  </span>
                </div>
              </div>
            {/each}

            <!-- New template button -->
            <button
              class="w-full flex items-center gap-2 justify-center px-3 py-2 text-sm rounded-lg text-gray-400 hover:text-gray-200 hover:bg-bg/50 cursor-pointer transition-colors"
              onclick={startNewTemplate}
            >
              <SvgPlus class="h-3.5 w-3.5" />
              New
            </button>
          </div>

          <!-- Template editor -->
          <div class="flex-1 flex flex-col min-w-0">
            {#if isNewTemplate}
              <div class="flex items-center gap-2 mb-2">
                <input
                  type="text"
                  bind:value={newTemplateName}
                  class="flex-1 rounded-lg bg-bg border border-border px-3 py-1.5 text-sm text-gray-200 focus:ring-2 focus:ring-accent focus:outline-none"
                  placeholder="Template name"
                />
                <button
                  class="px-3 py-1.5 text-xs rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 cursor-pointer"
                  onclick={cancelNewTemplate}
                >
                  Cancel
                </button>
              </div>
            {:else if selectedTemplateName}
              <div class="flex items-center justify-between mb-2">
                <h3 class="text-sm font-medium text-gray-300">
                  {selectedTemplateName}.jinja
                  {#if templateSource === "builtin"}
                    <span class="text-gray-500 text-xs ml-1">(built-in)</span>
                  {/if}
                </h3>
                {#if templateDirty}
                  <button
                    class="px-3 py-1.5 text-xs rounded-lg bg-accent hover:bg-accent-hover text-black font-medium cursor-pointer disabled:opacity-50"
                    onclick={saveTemplateContent}
                    disabled={isSavingTemplate}
                  >
                    {isSavingTemplate ? "Saving…" : "Save"}
                  </button>
                {/if}
              </div>
            {/if}

            {#if isLoadingTemplate}
              <div class="flex items-center justify-center py-12">
                <SvgSpinner class="h-6 w-6 animate-spin text-gray-400" />
              </div>
            {:else if selectedTemplateName || isNewTemplate}
              <div class="flex-1 min-h-0 overflow-hidden">
                <JinjaEditor
                  value={templateContent}
                  onchange={handleTemplateContentChange}
                />
              </div>
              {#if templateSource === "builtin" && !templateDirty}
                <p class="text-xs text-gray-500 mt-2">
                  Built-in template. Edit and save to create a user override.
                </p>
              {/if}
            {:else}
              <div class="flex items-center justify-center h-full text-gray-500 text-sm">
                Select or create a template
              </div>
            {/if}
          </div>
        </div>

        <!-- Delete template confirmation -->
        {#if confirmDeleteTemplate}
          <div class="p-4 rounded-lg bg-error/10 border border-error/20">
            <p class="text-sm text-gray-200 mb-3">
              Delete template <strong>{confirmDeleteTemplate}</strong>?
            </p>
            <div class="flex gap-2 justify-end">
              <button
                class="px-3 py-1.5 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 cursor-pointer"
                onclick={() => (confirmDeleteTemplate = null)}
              >
                Cancel
              </button>
              <button
                class="px-3 py-1.5 text-sm rounded-lg bg-error hover:bg-error/80 text-white cursor-pointer"
                onclick={() => handleDeleteTemplate(confirmDeleteTemplate!)}
              >
                Delete
              </button>
            </div>
          </div>
        {/if}

      <!-- ═══ Environments tab ═══ -->
      {:else if activeTab === "environments"}
        <div class="py-8 text-center">
          <p class="text-gray-400 text-sm mb-4">
            Environments store API connection settings (URL, token, default model).
          </p>
          <button
            class="px-4 py-2 text-sm rounded-lg bg-accent hover:bg-accent-hover text-black font-medium cursor-pointer"
            onclick={() => (showEnvManager = true)}
          >
            Open Environment Manager
          </button>
        </div>
      {/if}
    </div>
  </div>
</Dialog>
