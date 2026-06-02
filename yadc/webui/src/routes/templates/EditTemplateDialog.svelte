<script lang="ts">
  import Dialog from "$lib/components/ui/Dialog.svelte";
  import SvgClose from "$lib/icons/SvgClose.svelte";
  import SpinnerBlock from "$lib/components/ui/SpinnerBlock.svelte";
  import JinjaEditor from "$lib/components/ui/JinjaEditor.svelte";
  import { fetchTemplate, saveTemplate } from "$lib/stores/templates";

  interface Props {
    open: boolean;
    /** Existing template name, or null to create a new template. */
    templateName: string | null;
    onclose: () => void;
    /** Called after save with the template name. */
    onsaved: (name: string) => void;
  }

  let { open, templateName, onclose, onsaved }: Props = $props();

  let isNew = $derived(templateName === null);

  // New template state
  let newName = $state("");

  // Editor state
  let content = $state("");
  let templateSource: "user" | "builtin" | "" = $state("");
  let isLoading = $state(false);
  let isSaving = $state(false);
  let error: string | null = $state(null);

  $effect(() => {
    if (!open) return;
    error = null;
    isSaving = false;
    newName = "";
    content = "";
    templateSource = "";

    if (isNew) return;

    let cancelled = false;
    isLoading = true;

    (async () => {
      try {
        const info = await fetchTemplate(templateName!);
        if (cancelled) return;
        content = info.content;
        templateSource = info.source;
      } catch (e) {
        if (cancelled) return;
        error = e instanceof Error ? e.message : "Failed to load template";
      } finally {
        if (!cancelled) isLoading = false;
      }
    })();

    return () => {
      cancelled = true;
    };
  });

  async function handleSave() {
    const name = isNew ? newName.trim() : templateName!;
    if (!name) {
      error = "Template name is required";
      return;
    }

    isSaving = true;
    error = null;
    try {
      const info = await saveTemplate(name, content);
      onsaved(info.name);
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to save template";
    } finally {
      isSaving = false;
    }
  }
</script>

<Dialog
  class="dialog-panel max-w-3xl max-h-[85vh] overflow-hidden flex flex-col"
  {open}
  onclose={onclose}
>
  <div class="p-5 flex flex-col h-full max-h-[85vh]">
    <!-- Header -->
    <div class="dialog-header">
      <h2 class="dialog-title">{isNew ? "New Template" : `Edit ${templateName}`}</h2>
      <button class="btn-close" onclick={onclose}>
        <SvgClose class="h-5 w-5" />
      </button>
    </div>

    {#if error}
      <div class="alert-error mb-4">{error}</div>
    {/if}

    {#if isNew}
      <div class="mb-3">
        <label class="label" for="template-name">Template Name</label>
        <input
          id="template-name"
          type="text"
          bind:value={newName}
          class="input"
          placeholder="my-template"
        />
      </div>
    {:else if templateSource === "builtin"}
      <p class="text-xs text-gray-500 mb-3">
        Built-in template. Edit and save to create a user override.
      </p>
    {/if}

    {#if isLoading}
      <SpinnerBlock class="py-12" label="Loading template..." />
    {:else}
      <div class="flex-1 min-h-0 max-h-[55vh] overflow-y-auto" style="min-height: 200px;">
        <JinjaEditor value={content} onchange={(v) => (content = v)} />
      </div>
    {/if}

    <div class="btn-bar mt-4">
      <button class="btn-secondary" onclick={onclose} disabled={isSaving}>Cancel</button>
      <button class="btn-primary" onclick={handleSave} disabled={isSaving}>
        {isSaving ? "Saving..." : isNew ? "Create" : "Save"}
      </button>
    </div>
  </div>
</Dialog>
