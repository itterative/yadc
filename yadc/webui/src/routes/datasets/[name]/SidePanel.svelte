<script lang="ts">
  import CaptionSettings from "./CaptionSettings.svelte";
  import ImageDetail from "$lib/components/dataset/ImageDetail.svelte";
  import type { CaptionOptions } from "$lib/stores/captionOptions";
  import type { ImageInfo } from "$lib/stores/datasetImages";
  import SvgClose from "$lib/icons/SvgClose.svelte";
  import SvgMenuLeft from "$lib/icons/SvgMenuLeft.svelte";

  type PanelTab = "caption" | "details";

  interface Props {
    datasetName: string;
    focusedItem: ImageInfo | null;
    panelTab?: PanelTab;
    open?: boolean;
    captionOptions?: CaptionOptions;
    onstartcaptioning?: (options: CaptionOptions) => void;
    onpanelclose?: () => void;
    oncaptionupdated?: (imageId: number, caption: string) => void;
    oncaptionimage?: (imageId: number) => Promise<string>;
  }

  let {
    datasetName,
    focusedItem,
    panelTab = $bindable("caption"),
    open = $bindable(false),
    captionOptions = $bindable(),
    onstartcaptioning,
    onpanelclose,
    oncaptionupdated,
    oncaptionimage,
  }: Props = $props();
</script>

<!-- Mobile: backdrop -->
{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="lg:hidden fixed inset-0 bg-black/50 z-30 transition-opacity"
    onclick={() => { open = false; onpanelclose?.(); }}
  ></div>
{/if}

<!-- Side panel: desktop = inline, mobile = slide-over drawer -->
<div
  class="side-panel
    fixed inset-y-0 right-0 z-40 w-[80vw] max-w-[480px] shadow-2xl transition-transform duration-300
    lg:static lg:z-auto lg:w-[480px] lg:max-w-none lg:flex-shrink-0 lg:shadow-none lg:transition-none
    {open ? '' : 'translate-x-full'} lg:translate-x-0"
>
  <div class="flex flex-col h-full overflow-hidden bg-surface border-l border-border lg:border lg:rounded-xl">
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

      <button
        class="ml-auto mr-2 p-1 text-gray-400 hover:text-white transition-colors cursor-pointer lg:hidden"
        onclick={() => { open = false; onpanelclose?.(); }}
        title="Close panel"
        aria-label="Close panel"
      >
        <SvgClose class="w-5 h-5" />
      </button>
    </div>

    <!-- Tab content: always render both to preserve component state across tab switches -->
    <div class="flex-1 min-h-0 overflow-y-auto">
      <div class="{panelTab === 'caption' ? '' : 'hidden'}">
        <CaptionSettings
          {datasetName}
          bind:currentOptions={captionOptions}
          onstart={onstartcaptioning}
          onclose={() => { open = false; }}
        />
      </div>
      <div class="{panelTab === 'details' ? '' : 'hidden'}">
        {#if focusedItem !== null}
          <ImageDetail
            {datasetName}
            item={focusedItem}
            oncaptionupdated={oncaptionupdated}
            oncaptionimage={oncaptionimage}
          />
        {:else}
          <div class="flex items-center justify-center h-full text-gray-500 text-sm">
            Select an image to view details
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>

<!-- Mobile: floating toggle button -->
<button
  class="lg:hidden fixed bottom-6 right-6 z-20 w-16 h-16 rounded-full bg-accent text-white shadow-lg
         flex items-center justify-center hover:bg-accent-hover transition-colors cursor-pointer"
  onclick={() => (open = !open)}
  title="Toggle panel"
>
  <SvgMenuLeft class="w-6 h-6" />
</button>
