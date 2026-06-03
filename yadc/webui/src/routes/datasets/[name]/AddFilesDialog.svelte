<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import DatasetUploadPanel from '$lib/components/dataset/upload/DatasetUploadPanel.svelte';
    import type { DatasetInfo } from '$lib/stores/dataset';

    interface Props {
        open: boolean;
        datasetName: string;
        /** Files to pre-populate (e.g. from a drop event on the page). */
        files?: File[];
        onclose: () => void;
        oncomplete: (dataset: DatasetInfo) => void;
    }

    let { open, datasetName, files, onclose, oncomplete }: Props = $props();
</script>

<Dialog class="dialog-panel max-h-[85vh] max-w-2xl overflow-y-auto" {open} {onclose}>
    <div class="p-5">
        <div class="dialog-header">
            <h2 class="dialog-title">Add Files to {datasetName}</h2>
            <button class="btn-close" onclick={onclose}>
                <SvgClose class="h-5 w-5" />
            </button>
        </div>

        <DatasetUploadPanel
            mode="append"
            {datasetName}
            initialFiles={files}
            {oncomplete}
            {onclose}
        />
    </div>
</Dialog>
