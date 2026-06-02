<script lang="ts">
    import CaptionSettings from './CaptionSettings.svelte';
    import DatasetConfig from '$lib/components/datasets/DatasetConfig.svelte';
    import ImageDetail from '$lib/components/dataset/ImageDetail/ImageDetail.svelte';
    import PillTabs from '$lib/components/ui/tabs/PillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import type { ImageInfo } from '$lib/stores/datasetImages';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgMenuLeft from '$lib/icons/SvgMenuLeft.svelte';

    type PanelTab = 'caption' | 'details' | 'config' | 'history';

    interface Props {
        datasetName: string;
        source?: 'upload' | 'import' | 'create';
        focusedItem: ImageInfo | null;
        activeTab?: PanelTab;
        open?: boolean;
        onpanelclose?: () => void;
        onimagedelete?: () => void;
    }

    let {
        datasetName,
        source,
        focusedItem,
        activeTab: panelTab = $bindable('caption'),
        open = $bindable(false),
        onpanelclose,
        onimagedelete
    }: Props = $props();
</script>

<!-- Mobile: backdrop -->
{#if open}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
        class="fixed inset-0 z-30 bg-black/50 transition-opacity lg:hidden"
        onclick={() => {
            open = false;
            onpanelclose?.();
        }}
    ></div>
{/if}

<!-- Side panel: desktop = inline, mobile = slide-over drawer -->
<div
    class="side-panel
    fixed inset-y-0 right-0 z-40 w-[80vw] max-w-[480px] shadow-2xl transition-transform duration-300
    lg:static lg:w-[480px] lg:max-w-none lg:flex-shrink-0 lg:shadow-none lg:transition-none
    {open ? '' : 'translate-x-full'} lg:translate-x-0"
>
    <div
        class="flex h-full flex-col overflow-hidden border-l border-border bg-surface lg:rounded-xl lg:border"
    >
        <PillTabs bind:value={panelTab} class="h-full flex-1">
            {#snippet end()}
                <button
                    class="cursor-pointer p-1 text-gray-400 transition-colors hover:text-white lg:hidden"
                    onclick={() => {
                        open = false;
                        onpanelclose?.();
                    }}
                    title="Close panel"
                    aria-label="Close panel"
                >
                    <SvgClose class="h-5 w-5" />
                </button>
            {/snippet}
            <Tab id="caption" label="Caption" class="h-full overflow-y-auto">
                <CaptionSettings
                    {datasetName}
                    onclose={() => {
                        open = false;
                    }}
                />
            </Tab>
            <Tab id="details" label="Details" class="h-full overflow-y-auto">
                {#if focusedItem !== null}
                    <ImageDetail
                        {datasetName}
                        item={focusedItem}
                        {source}
                        ondelete={onimagedelete}
                    />
                {:else}
                    <div class="h-full text-center py-4 text-sm text-gray-500">
                        Select an image to view details
                    </div>
                {/if}
            </Tab>
            <Tab id="config" label="Config" class="h-full overflow-y-auto">
                <DatasetConfig {datasetName} {source} />
            </Tab>
        </PillTabs>
    </div>
</div>

<!-- Mobile: floating toggle button -->
<button
    class="fixed right-6 bottom-6 z-20 flex h-16 w-16 cursor-pointer items-center justify-center rounded-full
         bg-accent text-white shadow-lg transition-colors hover:bg-accent-hover lg:hidden"
    onclick={() => (open = !open)}
    title="Toggle panel"
>
    <SvgMenuLeft class="h-6 w-6" />
</button>
