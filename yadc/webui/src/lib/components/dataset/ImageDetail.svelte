<script lang="ts">
  import TomlEditor from "$lib/components/ui/TomlEditor.svelte";
  import SpinnerBlock from "$lib/components/ui/SpinnerBlock.svelte";
  import PromptPreview from "$lib/components/ui/PromptPreview.svelte";
  import {
    fetchCaption,
    mediaUrl,
    updateCaption,
    updateExtras,
    type CaptionData,
    type ImageInfo,
  } from "$lib/stores/datasetImages";

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

  let imgElement: HTMLImageElement | null = $state(null);

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

  function autosize(el: HTMLTextAreaElement) {
    function resize() {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 360) + "px";
    }
    resize();
    return { update: resize };
  }
</script>

{#if item !== null}
  <div class="flex flex-col h-full">
    <!-- Scrollable content -->
    <div class="flex-1 overflow-y-auto p-4 space-y-4">
      <!-- Filename -->
      <h2 class="dialog-title truncate text-base">{item.file_name}</h2>
      <p class="text-xs text-gray-500 truncate -mt-3">{item.path}</p>
      <!-- Image -->
      <div class="flex-shrink-0 flex items-start justify-center">
        <img
          src={imgSrc}
          alt={item.file_name}
          class="max-h-[50vh] w-auto rounded-lg bg-gray-500"
          width={item.width || undefined}
          height={item.height || undefined}
          bind:this={imgElement}
        />
      </div>

      <!-- Dimensions & badges -->
      <div class="flex items-center gap-3 flex-wrap">
        {#if item.width && item.height}
          <span class="text-sm text-gray-400">{item.width}×{item.height}</span>
        {/if}
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
          <SpinnerBlock size="h-4 w-4" label="Loading caption..." />
        {:else if captionError}
          <p class="text-sm text-error">{captionError}</p>
        {:else if isEditing}
          <div class="space-y-2">
            <textarea
              bind:value={editCaption}
              class="w-full rounded-lg bg-gray-800 p-3 text-sm text-gray-200 font-mono whitespace-pre-wrap resize-none max-h-60 focus:ring-2 focus:ring-accent focus:outline-none"
              placeholder="Enter caption..."
              oninput={(e) => {
                const el = e.currentTarget;
                el.style.height = 'auto';
                el.style.height = Math.min(el.scrollHeight, 360) + 'px';
              }}
              use:autosize
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
            <pre class="text-sm text-gray-200 bg-gray-800 rounded-lg p-3 whitespace-pre-wrap max-h-60 overflow-y-auto">{captionData.caption}</pre>
          {:else}
            <pre class="text-sm text-gray-500 italic bg-gray-800 rounded-lg p-3 whitespace-pre-wrap">No caption</pre>
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
                <pre class="text-sm text-gray-200 whitespace-pre-wrap font-mono max-h-40 overflow-y-auto">{text}</pre>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Prompt Preview -->
      <PromptPreview {datasetName} imageId={item.id} />


    </div>
  </div>
{/if}
