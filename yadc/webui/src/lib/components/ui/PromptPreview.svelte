<script lang="ts">
  import TomlEditor from "$lib/components/ui/TomlEditor.svelte";
  import {
    fetchPromptPreview,
    type PromptPreview as PromptPreviewData,
  } from "$lib/stores/datasetImages";
  import {
    fetchTemplates,
    type TemplateListItem,
  } from "$lib/stores/templates";

  interface Props {
    datasetName: string;
    imageId: number | null;
  }

  let { datasetName, imageId }: Props = $props();

  let promptPreview: PromptPreviewData | null = $state(null);
  let isLoadingPreview = $state(false);
  let previewError: string | null = $state(null);
  let previewTemplateName = $state("");
  let showPreview = $state(true);
  let templates: TemplateListItem[] = $state([]);
  let templatesLoaded = $state(false);

  // Load preview when image changes
  $effect(() => {
    const id = imageId;
    const dsName = datasetName;
    if (id === null) {
      promptPreview = null;
      previewError = null;
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        if (!templatesLoaded) {
          templates = await fetchTemplates();
          templatesLoaded = true;
        }
        const data = await fetchPromptPreview(dsName, id);
        if (cancelled) return;
        promptPreview = data;
        previewError = null;
      } catch (e) {
        if (cancelled) return;
        if (!templatesLoaded) templates = [];
        previewError = e instanceof Error ? e.message : "Failed to preview prompt";
      }
    })();

    return () => {
      cancelled = true;
    };
  });

  async function handlePreviewPrompt() {
    if (imageId === null) return;
    isLoadingPreview = true;
    previewError = null;
    promptPreview = null;
    showPreview = true;

    try {
      const opts = previewTemplateName
        ? { template_name: previewTemplateName }
        : {};
      promptPreview = await fetchPromptPreview(datasetName, imageId, opts);
    } catch (e) {
      previewError = e instanceof Error ? e.message : "Failed to preview prompt";
    } finally {
      isLoadingPreview = false;
    }
  }
</script>

<div class="pt-2">
  <div class="flex items-center gap-2 mb-2">
    <button
      class="text-sm text-accent hover:text-accent-hover cursor-pointer"
      onclick={() => (showPreview = !showPreview)}
    >
      {showPreview ? "▾" : "▸"} Preview Prompt
    </button>
  </div>

  {#if showPreview}
    <div class="space-y-3">
      <div class="flex gap-2 items-end">
        <div class="flex-1">
          <label class="label mb-1" for="preview-template">Template</label>
          <select
            id="preview-template"
            bind:value={previewTemplateName}
            class="input-sm"
            onchange={() => handlePreviewPrompt()}
          >
            <option value="">(default)</option>
            {#each templates as t (t.name)}
              <option value={t.name}>{t.name} {t.source === "builtin" ? "(built-in)" : ""}</option>
            {/each}
          </select>
        </div>
        <button
          class="btn-primary px-3 py-1.5"
          onclick={handlePreviewPrompt}
          disabled={isLoadingPreview}
        >
          {isLoadingPreview ? "Loading..." : "Render"}
        </button>
      </div>

      {#if previewError}
        <p class="text-sm text-error">{previewError}</p>
      {/if}

      {#if promptPreview}
        {@const ctxEntries = Object.entries(promptPreview.template_context).filter(
          ([k]) => k !== "caption_suffix" && k !== "toml_suffix" && k !== "history_suffix",
        )}
        <div>
          <h4 class="text-xs font-medium text-gray-400 mb-1">System Prompt</h4>
          <pre class="text-sm text-gray-200 bg-gray-800 rounded-lg p-3 whitespace-pre-wrap max-h-40 overflow-y-auto">{promptPreview.system_prompt}</pre>
        </div>
        <div>
          <h4 class="text-xs font-medium text-gray-400 mb-1">User Prompt</h4>
          <pre class="text-sm text-gray-200 bg-gray-800 rounded-lg p-3 whitespace-pre-wrap max-h-60 overflow-y-auto">{promptPreview.user_prompt}</pre>
        </div>
        <details class="group">
          <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-300">Template context ({ctxEntries.length} variables)</summary>
          <div class="mt-1">
            <TomlEditor
              value={promptPreview.template_context_toml}
              editable={false}
            />
          </div>
        </details>
      {/if}
    </div>
  {/if}
</div>
