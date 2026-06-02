<script lang="ts">
  import JinjaEditor from "$lib/components/ui/JinjaEditor.svelte";
  import ConfirmDelete from "$lib/components/ui/ConfirmDelete.svelte";
  import SpinnerBlock from "$lib/components/ui/SpinnerBlock.svelte";
  import SvgDelete from "$lib/icons/SvgDelete.svelte";
  import SvgPlus from "$lib/icons/SvgPlus.svelte";
  import {
    fetchTemplates,
    fetchTemplate,
    saveTemplate,
    deleteTemplate,
    type TemplateListItem,
  } from "$lib/stores/templates";

  interface Props {
    /** Reload data when this becomes true (e.g. dialog opens). */
    open: boolean;
  }

  let { open }: Props = $props();

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

  $effect(() => {
    if (open) {
      templateError = null;
      templateDirty = false;
      confirmDeleteTemplate = null;
      isNewTemplate = false;
      loadTemplates();
    }
  });

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
</script>

{#if templateError}
  <div class="alert-error">{templateError}</div>
{/if}

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
          class="input-sm"
          placeholder="Template name"
        />
        <button
          class="btn-secondary px-3 py-1.5 text-xs"
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
            class="btn-primary px-3 py-1.5 text-xs"
            onclick={saveTemplateContent}
            disabled={isSavingTemplate}
          >
            {isSavingTemplate ? "Saving…" : "Save"}
          </button>
        {/if}
      </div>
    {/if}

    {#if isLoadingTemplate}
      <SpinnerBlock class="py-12" />
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

<ConfirmDelete
  open={confirmDeleteTemplate !== null}
  oncancel={() => (confirmDeleteTemplate = null)}
  onconfirm={() => handleDeleteTemplate(confirmDeleteTemplate!)}
>
  Delete template <strong>{confirmDeleteTemplate}</strong>?
</ConfirmDelete>
