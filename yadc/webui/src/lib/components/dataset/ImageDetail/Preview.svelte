<script lang="ts">
    import PromptPreview from '$lib/components/ui/PromptPreview.svelte';
    import type { ImageInfo } from '$lib/stores/datasetImages';

    interface Props {
        datasetName: string;
        item: ImageInfo;
    }

    let { datasetName, item }: Props = $props();
</script>

<PromptPreview {datasetName} imageId={item.id} />
