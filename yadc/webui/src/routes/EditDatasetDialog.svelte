<script lang="ts">
  import Dialog from "$lib/components/ui/Dialog.svelte";
  import SvgClose from "$lib/icons/SvgClose.svelte";
  import SpinnerBlock from "$lib/components/ui/SpinnerBlock.svelte";
  import TomlEditor from "$lib/components/ui/TomlEditor.svelte";
  import { fetchConfig, updateConfig } from "$lib/stores/configs";

  interface Props {
    open: boolean;
    datasetName: string;
    onclose: () => void;
    onsaved: () => void;
  }

  let { open, datasetName, onclose, onsaved }: Props = $props();

  let rawContent = $state("");
  let isLoading = $state(false);
  let isSaving = $state(false);
  let error: string | null = $state(null);

  $effect(() => {
    if (!open) return;
    let cancelled = false;
    isLoading = true;
    error = null;
    rawContent = "";

    (async () => {
      try {
        const config = await fetchConfig(datasetName);
        if (cancelled) return;
        rawContent = config.content;
      } catch (e) {
        if (cancelled) return;
        error = e instanceof Error ? e.message : "Failed to load config";
      } finally {
        if (!cancelled) isLoading = false;
      }
    })();

    return () => {
      cancelled = true;
    };
  });

  async function handleSave() {
    isSaving = true;
    error = null;
    try {
      await updateConfig(datasetName, rawContent);
      onsaved();
      onclose();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to save config";
    } finally {
      isSaving = false;
    }
  }
</script>

<Dialog
  class="dialog-panel max-w-lg max-h-[80vh] overflow-y-auto"
  {open}
  {onclose}
>
  <div class="p-5">
    <!-- Header -->
    <div class="dialog-header">
      <h2 class="dialog-title">Edit {datasetName}</h2>
      <button class="btn-close" onclick={onclose}>
        <SvgClose class="h-5 w-5" />
      </button>
    </div>

    {#if error}
      <div class="alert-error mb-4">{error}</div>
    {/if}

    {#if isLoading}
      <SpinnerBlock class="py-4" size="h-5 w-5" label="Loading config..." />
    {:else}
      <div class="space-y-4">
        <div class="min-h-[200px] max-h-[50vh] overflow-y-auto">
          <TomlEditor value={rawContent} editable={true} onchange={(v) => (rawContent = v)} />
        </div>

        <div class="btn-bar">
          <button class="btn-secondary" onclick={onclose} disabled={isSaving}>Cancel</button>
          <button class="btn-primary" onclick={handleSave} disabled={isSaving}>
            {isSaving ? "Saving..." : "Save & Rescan"}
          </button>
        </div>
      </div>
    {/if}
  </div>
</Dialog>
