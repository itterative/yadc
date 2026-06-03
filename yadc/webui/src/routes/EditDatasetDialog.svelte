<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import PillTabs from '$lib/components/ui/tabs/PillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import DatasetConfig from '$lib/components/datasets/DatasetConfig.svelte';
    import DatasetManageTab from '$lib/components/datasets/DatasetManageTab.svelte';
    import DatasetUploadPanel from '$lib/components/datasets/DatasetUploadPanel.svelte';

    interface Props {
        open: boolean;
        datasetName: string;
        source?: 'upload' | 'import' | 'create';
        onclose: () => void;
        onsaved: () => void;
    }

    let { open, datasetName, source, onclose, onsaved }: Props = $props();

    let mode: 'config' | 'manage' | 'upload' = $state('config');

    function handleConfigSaved() {
        onsaved();
    }

    function handleUploadComplete() {
        onsaved();
        onclose();
    }
</script>

<Dialog class="dialog-panel h-[55vh] max-w-2xl max-sm:h-[90vh]" {open} {onclose}>
    <div class="grid h-full grid-rows-[min-content_minmax(0,1fr)] p-5">
        <div class="dialog-header shrink-0">
            <h2 class="dialog-title">Edit {datasetName}</h2>
            <button class="btn-close" onclick={onclose}>
                <SvgClose class="h-5 w-5" />
            </button>
        </div>

        <PillTabs bind:value={mode} hideSingle>
            <Tab id="config" label="Config" class="h-full">
                <DatasetConfig {datasetName} {source} onsaved={handleConfigSaved} />
            </Tab>
            <Tab id="manage" label="Manage" class="h-full overflow-y-auto">
                <DatasetManageTab {datasetName} {source} onchanged={handleConfigSaved} />
            </Tab>
            {#if source === 'upload'}
                <Tab id="upload" label="Upload" class="h-full overflow-y-auto">
                    <DatasetUploadPanel
                        mode="append"
                        {datasetName}
                        oncomplete={handleUploadComplete}
                        {onclose}
                    />
                </Tab>
            {/if}
        </PillTabs>
    </div>
</Dialog>
