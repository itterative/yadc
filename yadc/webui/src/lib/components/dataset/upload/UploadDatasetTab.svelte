<script lang="ts">
    import DatasetUploadPanel from './DatasetUploadPanel.svelte';
    import type { DatasetInfo } from '$lib/stores/dataset';

    interface Props {
        oncreated: (dataset: DatasetInfo) => void;
        onclose: () => void;
    }

    let { oncreated, onclose }: Props = $props();
</script>

<DatasetUploadPanel mode="create" oncomplete={oncreated} {onclose} />
