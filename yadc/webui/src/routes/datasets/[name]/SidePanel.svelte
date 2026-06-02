<script lang="ts">
  import CaptionSettings from "./CaptionSettings.svelte";
  import ImageDetail from "$lib/components/dataset/ImageDetail.svelte";
  import type { CaptionOptions } from "$lib/stores/captionOptions";
  import type { ImageInfo } from "$lib/stores/datasetImages";

  type PanelTab = "caption" | "details";

  interface Props {
    datasetName: string;
    focusedItem: ImageInfo | null;
    isMobile: boolean;
    panelTab?: PanelTab;
    open?: boolean;
    onstartcaptioning?: (options: CaptionOptions) => void;
    onpanelclose?: () => void;
    oncaptionupdated?: (imageId: number, caption: string) => void;
  }

  let {
    datasetName,
    focusedItem,
    isMobile,
    panelTab = $bindable("caption"),
    open = $bindable(false),
    onstartcaptioning,
    onpanelclose,
    oncaptionupdated,
  }: Props = $props();

  function handleClose() {
    onpanelclose?.();
  }
</script>

<!-- Mobile: backdrop -->
{#if isMobile && open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="lg:hidden fixed inset-0 bg-black/50 z-30 transition-opacity"
    onclick={() => (open = false)}
  ></div>
{/if}

<!-- Side panel: desktop = inline, mobile = slide-over drawer -->
<div
  class="side-panel {isMobile
    ? 'fixed inset-y-0 right-0 z-40 w-[80vw] max-w-[480px] shadow-2xl transition-transform duration-300'
    : 'w-[480px] flex-shrink-0'
  }"
  style:transform={isMobile ? (open ? 'translateX(0)' : 'translateX(100%)') : undefined}
>
  <div class="flex flex-col h-full overflow-hidden bg-surface {isMobile ? 'border-l border-border' : 'rounded-xl border border-border'}">
    <!-- Tab bar -->
    <div class="flex items-center border-b border-border flex-shrink-0">
      <button
        class="px-4 py-2.5 text-sm font-medium transition-colors cursor-pointer border-b-2 {panelTab === 'caption'
          ? 'text-white border-accent'
          : 'text-gray-400 border-transparent hover:text-gray-200'
        }"
        onclick={() => (panelTab = "caption")}
      >
        Caption
      </button>
      <button
        class="px-4 py-2.5 text-sm font-medium transition-colors cursor-pointer border-b-2 {panelTab === 'details'
          ? 'text-white border-accent'
          : 'text-gray-400 border-transparent hover:text-gray-200'
        }"
        onclick={() => (panelTab = "details")}
      >
        Details
      </button>

      {#if isMobile}
        <button
          class="ml-auto mr-2 p-1 text-gray-400 hover:text-white transition-colors cursor-pointer"
          onclick={() => (open = false)}
          title="Close panel"
          aria-label="Close panel"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      {/if}
    </div>

    <!-- Tab content -->
    <div class="flex-1 min-h-0 overflow-y-auto">
      {#if panelTab === "caption"}
        <CaptionSettings
          {datasetName}
          onstart={onstartcaptioning}
          onclose={() => { if (isMobile) open = false; }}
        />
      {:else if panelTab === "details"}
        {#if focusedItem !== null}
          <ImageDetail
            {datasetName}
            item={focusedItem}
            onclose={handleClose}
            oncaptionupdated={oncaptionupdated}
          />
        {:else}
          <div class="flex items-center justify-center h-full text-gray-500 text-sm">
            Select an image to view details
          </div>
        {/if}
      {/if}
    </div>
  </div>
</div>

<!-- Mobile: floating toggle button -->
{#if isMobile}
  <button
    class="fixed bottom-6 right-6 z-20 w-16 h-16 rounded-full bg-accent text-white shadow-lg
           flex items-center justify-center hover:bg-accent-hover transition-colors cursor-pointer"
    onclick={() => (open = !open)}
    title="Toggle panel"
  >
    {#if panelTab === 'caption'}
      <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
      </svg>
    {:else}
      <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    {/if}
  </button>
{/if}
