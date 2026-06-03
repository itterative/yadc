<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import PillTabs from '$lib/components/ui/tabs/PillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import UploadDatasetTab from '$lib/components/dataset/upload/UploadDatasetTab.svelte';
    import CreateDatasetTab from '$lib/components/dataset/upload/CreateDatasetTab.svelte';
    import type { DatasetInfo } from '$lib/stores/datasetImages';

    interface Props {
        open: boolean;
        onclose: () => void;
        oncreated: (dataset: DatasetInfo) => void;
    }

    let { open, onclose, oncreated }: Props = $props();

    let mode: 'upload' | 'create' = $state('create');

    function handleCreated(dataset: DatasetInfo) {
        mode = 'create';
        oncreated(dataset);
    }
</script>

<Dialog class="dialog-panel max-h-[80vh] max-w-lg overflow-y-auto" {open} {onclose}>
    <div class="p-5">
        <div class="dialog-header">
            <h2 class="dialog-title">Add Dataset</h2>
            <button class="btn-close" onclick={onclose}>
                <SvgClose class="h-5 w-5" />
            </button>
        </div>

        <PillTabs bind:value={mode}>
            <Tab id="upload" label="Upload">
                <UploadDatasetTab oncreated={handleCreated} {onclose} />
            </Tab>
            <Tab id="create" label="Create">
                <CreateDatasetTab oncreated={handleCreated} {onclose} />
            </Tab>
        </PillTabs>
    </div>
</Dialog>
