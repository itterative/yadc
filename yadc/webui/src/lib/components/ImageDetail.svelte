<script lang="ts">
  import Dialog from "$lib/components/Dialog.svelte";
  import SvgSpinner from "$lib/icons/SvgSpinner.svelte";
  import TomlEditor from "$lib/components/TomlEditor.svelte";
  import {
    fetchCaption,
    fetchPromptPreview,
    mediaUrl,
    updateCaption,
    updateExtras,
    type CaptionData,
    type ImageInfo,
    type PromptPreview,
  } from "$lib/stores/datasetImages";

  import {
    fetchTemplates,
    type TemplateListItem,
  } from "$lib/stores/templates";

  interface Props {
    datasetName: string;
    item: ImageInfo | null;
    onclose: () => void;
    oncaptionupdated?: (imageId: number, caption: string) => void;
  }

  let { datasetName, item, onclose, oncaptionupdated }: Props = $props();

  let captionData: CaptionData | null = $state(null);
  let isLoadingCaption = $state(false);
  let captionError: string | null = $state(null);
  let isEditing = $state(false);
  let editCaption = $state("");
  let isSaving = $state(false);

  // TOML extras editing state
  let isEditingExtras = $state(false);
  let editExtrasRaw = $state("");

  let dialogOpen = $state(false);
  let imgElement: HTMLImageElement | null = $state(null);

  // Prompt preview state
  let promptPreview: PromptPreview | null = $state(null);
  let isLoadingPreview = $state(false);
  let previewError: string | null = $state(null);
  let previewTemplateName = $state("");
  let showPreview = $state(false);
  let templates: TemplateListItem[] = $state([]);
  let templatesLoaded = $state(false);

  $effect(() => {
    dialogOpen = item !== null;
  });

  // Load caption when item changes
  $effect(() => {
    if (item === null) {
      captionData = null;
      captionError = null;
      isEditing = false;
      isEditingExtras = false;
      return;
    }

    let cancelled = false;
    isLoadingCaption = true;
    captionError = null;
    captionData = null;
    isEditing = false;
    isEditingExtras = false;

    (async () => {
      try {
        const data = await fetchCaption(datasetName, item.id);
        if (cancelled) return;
        captionData = data;
        editCaption = data.caption || "";
      } catch (e) {
        if (cancelled) return;
        captionError = e instanceof Error ? e.message : "Failed to load caption";
      } finally {
        if (!cancelled) isLoadingCaption = false;
      }
    })();

    return () => {
      cancelled = true;
    };
  });

  let aspectRatio = $derived.by(() => {
    if (!item?.width || !item?.height) return undefined;
    return `${item.width} / ${item.height}`;
  });

  let imgSrc = $derived(
    item !== null
      ? mediaUrl(datasetName, item.id)
      : "",
  );

  async function handleSave() {
    if (item === null || !isEditing) return;
    isSaving = true;
    try {
      await updateCaption(datasetName, item.id, editCaption);
      captionData = { ...captionData!, caption: editCaption };
      isEditing = false;
      oncaptionupdated?.(item.id, editCaption);
    } catch (e) {
      captionError = e instanceof Error ? e.message : "Failed to save caption";
    } finally {
      isSaving = false;
    }
  }

  function handleCancelEdit() {
    isEditing = false;
    editCaption = captionData?.caption || "";
  }

  let isSavingExtras = $state(false);

  async function handleSaveExtras() {
    if (item === null || !isEditingExtras) return;
    isSavingExtras = true;
    try {
      await updateExtras(datasetName, item.id, editExtrasRaw);
      captionData = { ...captionData!, extras_raw: editExtrasRaw };
      isEditingExtras = false;
    } catch (e) {
      captionError = e instanceof Error ? e.message : "Failed to save extras";
    } finally {
      isSavingExtras = false;
    }
  }

  function handleCancelEditExtras() {
    isEditingExtras = false;
    editExtrasRaw = captionData?.extras_raw || "";
  }

  async function handleTogglePreview() {
    showPreview = !showPreview;
    if (showPreview && !templatesLoaded) {
      try {
        templates = await fetchTemplates();
      } catch {
        templates = [];
      }
      templatesLoaded = true;
    }
  }

  async function handlePreviewPrompt() {
    if (item === null) return;
    isLoadingPreview = true;
    previewError = null;
    promptPreview = null;
    showPreview = true;

    try {
      const opts = previewTemplateName
        ? { template_name: previewTemplateName }
        : {};
      promptPreview = await fetchPromptPreview(datasetName, item.id, opts);
    } catch (e) {
      previewError = e instanceof Error ? e.message : "Failed to preview prompt";
    } finally {
      isLoadingPreview = false;
    }
  }
</script>

<Dialog class="max-w-5xl w-full max-h-[90vh] overflow-y-auto m-auto" open={dialogOpen} {onclose}>
  {#if item !== null}
    <div class="flex flex-col lg:flex-row gap-4 p-2">
      <!-- Image -->
      <div class="flex-shrink-0 flex items-start justify-center lg:w-1/2">
        <img
          src={imgSrc}
          alt={item.file_name}
          class="max-h-[70vh] w-auto rounded-lg bg-gray-500"
          width={item.width || undefined}
          height={item.height || undefined}
          style:aspect-ratio={aspectRatio}
          bind:this={imgElement}
        />
      </div>

      <!-- Info panel -->
      <div class="flex-1 min-w-0 space-y-4">
        <div>
          <h2 class="dialog-title truncate">{item.file_name}</h2>
          <p class="text-sm text-gray-400">
            {#if item.width && item.height}
              {item.width}×{item.height}
            {/if}
          </p>
        </div>

        <!-- Caption -->
        <div>
          <div class="flex items-center justify-between mb-2">
            <h3 class="text-sm font-medium text-gray-300">Caption</h3>
            {#if captionData && !isEditing}
              <button
                class="text-xs text-accent hover:text-accent-hover cursor-pointer"
                onclick={() => {
                  editCaption = captionData?.caption || "";
                  isEditing = true;
                }}
              >
                Edit
              </button>
            {/if}
          </div>

          {#if isLoadingCaption}
            <div class="flex items-center gap-2 text-gray-400">
              <SvgSpinner class="h-4 w-4 animate-spin" />
              <span class="text-sm">Loading caption...</span>
            </div>
          {:else if captionError}
            <p class="text-sm text-error">{captionError}</p>
          {:else if isEditing}
            <div class="space-y-2">
              <textarea
                bind:value={editCaption}
                class="w-full rounded-lg bg-gray-800 border border-gray-600 p-3 text-sm text-gray-200 resize-y min-h-[120px] focus:ring-2 focus:ring-accent focus:outline-none"
                placeholder="Enter caption..."
              ></textarea>
              <div class="btn-bar">
                <button
                  class="btn-secondary px-3 py-1.5"
                  onclick={handleCancelEdit}
                  disabled={isSaving}
                >
                  Cancel
                </button>
                <button
                  class="btn-primary px-3 py-1.5"
                  onclick={handleSave}
                  disabled={isSaving}
                >
                  {isSaving ? "Saving..." : "Save"}
                </button>
              </div>
            </div>
          {:else if captionData}
            {#if captionData.caption}
              <p class="text-sm text-gray-200 whitespace-pre-wrap">{captionData.caption}</p>
            {:else}
              <p class="text-sm text-gray-500 italic">No caption</p>
            {/if}
          {/if}
        </div>

        <!-- TOML extras -->
        {#if captionData?.extras_raw}
          <div>
            <div class="flex items-center justify-between mb-2">
              <h3 class="text-sm font-medium text-gray-300">TOML Extras</h3>
              {#if !isEditingExtras}
                <button
                  class="text-xs text-accent hover:text-accent-hover cursor-pointer"
                  onclick={() => {
                    editExtrasRaw = captionData?.extras_raw || "";
                    isEditingExtras = true;
                  }}
                >
                  Edit
                </button>
              {:else}
                <div class="flex gap-2">
                  <button
                  class="btn-secondary px-2 py-1 text-xs"
                    onclick={handleCancelEditExtras}
                    disabled={isSavingExtras}
                  >
                    Cancel
                  </button>
                  <button
                    class="btn-primary px-2 py-1 text-xs"
                    onclick={handleSaveExtras}
                    disabled={isSavingExtras}
                  >
                    {isSavingExtras ? "Saving..." : "Save"}
                  </button>
                </div>
              {/if}
            </div>
            <TomlEditor
              value={isEditingExtras ? editExtrasRaw : captionData.extras_raw}
              editable={isEditingExtras}
              onchange={(v) => (editExtrasRaw = v)}
            />
          </div>
        {:else if captionData?.extras && Object.keys(captionData.extras).length > 0}
          <div>
            <h3 class="text-sm font-medium text-gray-300 mb-2">TOML Extras</h3>
            <pre class="text-xs text-gray-400 bg-gray-800 rounded-lg p-3 overflow-x-auto">{JSON.stringify(captionData.extras, null, 2)}</pre>
          </div>
        {/if}

        <!-- Drafts -->
        {#if captionData?.drafts && Object.keys(captionData.drafts).length > 0}
          <div>
            <h3 class="text-sm font-medium text-gray-300 mb-2">Drafts</h3>
            <div class="space-y-2">
              {#each Object.entries(captionData.drafts) as [name, text] (name)}
                <div class="bg-gray-800 rounded-lg p-3">
                  <p class="text-xs font-medium text-accent mb-1">{name}</p>
                  <p class="text-sm text-gray-200 whitespace-pre-wrap">{text}</p>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Prompt Preview -->
        <div class="pt-2">
          <div class="flex items-center gap-2 mb-2">
            <button
              class="text-sm text-accent hover:text-accent-hover cursor-pointer"
              onclick={handleTogglePreview}
            >
              {showPreview ? "▾" : "▸"} Preview Prompt
            </button>
          </div>

          {#if showPreview}
            <div class="space-y-3">
              <div class="flex gap-2 items-end">
                <div class="flex-1">
                  <label class="label mb-1">Template</label>
                  <select
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
                  <pre class="text-xs text-gray-400 bg-gray-800 rounded-lg p-3 mt-1 overflow-x-auto">{JSON.stringify(Object.fromEntries(ctxEntries), null, 2)}</pre>
                </details>
              {/if}
            </div>
          {/if}
        </div>

        <!-- Status badges -->
        <div class="flex gap-2 pt-2">
          {#if item.has_caption}
            <span class="badge-success rounded-full px-2 py-1">Captioned</span>
          {:else}
            <span class="badge-muted rounded-full px-2 py-1">No caption</span>
          {/if}
          {#if item.has_toml}
            <span class="badge-accent rounded-full px-2 py-1">TOML</span>
          {/if}
          {#if item.draft_names.length > 0}
            <span class="badge-error rounded-full px-2 py-1">
              {item.draft_names.length} draft{item.draft_names.length !== 1 ? "s" : ""}
            </span>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</Dialog>
