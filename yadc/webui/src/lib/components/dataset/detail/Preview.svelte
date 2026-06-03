<script lang="ts">
    import PromptPreview from '$lib/components/ui/PromptPreview.svelte';
    import type { CaptionData, ImageInfo } from '$lib/stores/dataset';

    interface Props {
        datasetName: string;
        captionData: CaptionData | null;
        item: ImageInfo;
    }

    let { datasetName, captionData, item }: Props = $props();
</script>

<PromptPreview {datasetName} {captionData} imageId={item.id} />
