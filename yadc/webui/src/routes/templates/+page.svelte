<script lang="ts">
  import { onMount } from "svelte";
  import { browser } from "$app/environment";
  import {
    fetchTemplates,
    deleteTemplate,
    type TemplateListItem,
  } from "$lib/stores/templates";
  import EditTemplateDialog from "./EditTemplateDialog.svelte";
  import ConfirmDelete from "$lib/components/ui/ConfirmDelete.svelte";
  import SvgDelete from "$lib/icons/SvgDelete.svelte";
  import SvgEdit from "$lib/icons/SvgEdit.svelte";
  import SvgFile from "$lib/icons/SvgFile.svelte";
  import SvgPlus from "$lib/icons/SvgPlus.svelte";

  let templates: TemplateListItem[] = $state([]);
  let loading = $state(true);
  let error = $state("");
  let showAddTemplate = $state(false);

  // Delete state
  let deletingTemplate: TemplateListItem | null = $state(null);

  // Edit state
  let editingTemplate: TemplateListItem | null = $state(null);

  onMount(() => {
    if (!browser) return;
    loadTemplates();
  });

  async function loadTemplates() {
    try {
      templates = await fetchTemplates();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load templates";
    } finally {
      loading = false;
    }
  }

  async function handleDeleteConfirm() {
    if (!deletingTemplate) return;
    try {
      await deleteTemplate(deletingTemplate.name);
      templates = templates.filter((t) => t.name !== deletingTemplate!.name);
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to delete template";
    } finally {
      deletingTemplate = null;
    }
  }

  function handleTemplateSaved() {
    loadTemplates();
    editingTemplate = null;
  }

  function handleTemplateCreated(name: string) {
    loadTemplates();
    showAddTemplate = false;
    // Open the new template for editing
    editingTemplate = { name, source: "user" };
  }
</script>

<h1 class="text-xl font-bold mb-6">Templates</h1>

{#if loading}
  <p class="text-muted">Loading templates...</p>
{:else if error}
  <p class="text-error">Error: {error}</p>
{:else if templates.length === 0}
  <div class="empty-state py-12">
    <h2>No templates found</h2>
    <p>Create a new Jinja2 template to get started.</p>
    <button class="btn-primary mt-4" onclick={() => (showAddTemplate = true)}>
      Add Template
    </button>
  </div>
{:else}
  <div class="grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-4">
    {#each templates as template (template.name)}
      <div
        class="card group relative cursor-pointer hover:opacity-90 transition-opacity"
        onclick={() => (editingTemplate = template)}
        onkeydown={(e) => { if (e.key === "Enter" || e.key === " ") editingTemplate = template; }}
        role="button"
        tabindex="0"
      >
        <div class="flex items-center justify-center aspect-video w-full bg-border/30">
          <SvgFile class="w-8 h-8 text-muted opacity-50" />
        </div>
        <div class="card-body">
          <h3 class="text-fg mb-1">{template.name}</h3>
          <span class="badge {template.source === 'builtin' ? 'badge-muted' : 'badge-accent'}">
            {template.source === "builtin" ? "Built-in" : "User"}
          </span>
        </div>

        <!-- Action buttons (top-right corner) -->
        <div class="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            class="p-1.5 rounded-md bg-black/60 hover:bg-black/80 text-gray-300 hover:text-white cursor-pointer"
            title="Edit template"
            onclick={(e) => { e.preventDefault(); e.stopPropagation(); editingTemplate = template; }}
          >
            <SvgEdit class="w-4 h-4" />
          </button>
          {#if template.source === "user"}
            <button
              class="p-1.5 rounded-md bg-black/60 hover:bg-black/80 text-gray-300 hover:text-error cursor-pointer"
              title="Delete template"
              onclick={(e) => { e.preventDefault(); e.stopPropagation(); deletingTemplate = template; }}
            >
              <SvgDelete class="w-4 h-4" />
            </button>
          {/if}
        </div>
      </div>
    {/each}
    <button
      class="card border-dashed border-gray-600 hover:border-accent flex items-center justify-center cursor-pointer min-h-56"
      onclick={() => (showAddTemplate = true)}
      title="Add template"
    >
      <span class="text-gray-400 text-sm flex items-center gap-2">
        <SvgPlus class="w-4 h-4" />
        Add Template
      </span>
    </button>
  </div>

  <!-- Delete confirmation -->
  {#if deletingTemplate}
    <ConfirmDelete
      open={true}
      oncancel={() => (deletingTemplate = null)}
      onconfirm={handleDeleteConfirm}
    >
      Delete template "{deletingTemplate.name}"? This action cannot be undone.
    </ConfirmDelete>
  {/if}
{/if}

<!-- Edit dialog -->
{#if editingTemplate}
  <EditTemplateDialog
    open={true}
    templateName={editingTemplate.name}
    onclose={() => (editingTemplate = null)}
    onsaved={handleTemplateSaved}
  />
{/if}

<!-- Add dialog -->
<EditTemplateDialog
  open={showAddTemplate}
  templateName={null}
  onclose={() => (showAddTemplate = false)}
  onsaved={(name) => handleTemplateCreated(name)}
/>
