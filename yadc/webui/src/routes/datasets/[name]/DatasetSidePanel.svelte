<script lang="ts">
    import CaptionSettingsPanel from '$lib/components/caption/CaptionSettingsPanel.svelte';
    import TagSettingsPanel from '$lib/components/tagging/TagSettingsPanel.svelte';
    import DatasetConfig from '$lib/components/dataset/config/DatasetConfig.svelte';
    import ImageDetail from '$lib/components/dataset/detail/ImageDetail.svelte';
    import PillTabs from '$lib/components/ui/tabs/PillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import SidePanel from '$lib/components/ui/SidePanel.svelte';
    import type { ImageInfo } from '$lib/stores/dataset';
    import SvgClose from '$lib/icons/SvgClose.svelte';

    type PanelTab = 'caption' | 'tags' | 'details' | 'config';

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

    function handleCloseClick() {
        // Mobile-only X inside the tab bar. Mutating ``open`` propagates
        // through ``bind:open`` to the parent, and the parent's
        // ``onpanelclose`` (wired as the generic SidePanel's ``onclose``
        // by callers) handles the rest.
        open = false;
        onpanelclose?.();
    }
</script>

<SidePanel bind:open onclose={onpanelclose}>
    <PillTabs bind:value={panelTab} class="h-full flex-1">
        {#snippet end()}
            <button
                class="cursor-pointer p-1 text-gray-400 transition-colors hover:text-white lg:hidden"
                onclick={handleCloseClick}
                title="Close panel"
                aria-label="Close panel"
            >
                <SvgClose class="h-5 w-5" />
            </button>
        {/snippet}
        <Tab id="caption" label="Caption" class="h-full overflow-y-auto">
            <CaptionSettingsPanel
                {datasetName}
                onclose={() => {
                    open = false;
                }}
            />
        </Tab>
        <Tab id="tags" label="Tags" class="h-full overflow-y-auto">
            <TagSettingsPanel
                {datasetName}
                onclose={() => {
                    open = false;
                }}
            />
        </Tab>
        <Tab id="details" label="Details" class="h-full overflow-y-auto">
            {#if focusedItem !== null}
                <ImageDetail {datasetName} item={focusedItem} {source} ondelete={onimagedelete} />
            {:else}
                <div class="h-full py-4 text-center text-sm text-gray-500">
                    Select an image to view details
                </div>
            {/if}
        </Tab>
        <Tab id="config" label="Config" class="h-full overflow-y-auto">
            <DatasetConfig {datasetName} {source} />
        </Tab>
    </PillTabs>
</SidePanel>
