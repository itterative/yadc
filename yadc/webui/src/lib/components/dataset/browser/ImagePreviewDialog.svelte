<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import { mediaUrl, type ImageInfo } from '$lib/stores/dataset';

    interface Props {
        open: boolean;
        onclose: () => void;
        datasetName: string;
        item: ImageInfo | null;
    }

    let { open, onclose, datasetName, item }: Props = $props();

    let imgSrc = $derived(item !== null ? mediaUrl(datasetName, item.id) : '');

    let aspectRatio = $derived.by(() => {
        if (!item) {
            return undefined;
        }

        if (!item.width || !item.height) {
            return undefined;
        }

        return `${item.width} / ${item.height}`;
    });
</script>

<Dialog class="relative m-auto max-h-[calc(90vh-4rem)] max-w-[90vw] flex flex-col items-center" {open} {onclose}>
    {#if item !== null}
        <img
            src={imgSrc}
            alt={item.file_name}
            class="max-h-[inherit] w-full flex-1 rounded-xl object-contain shadow-2xl"
            style:aspect-ratio={aspectRatio}
            width={item.width || undefined}
            height={item.height || undefined}
        />
        <p
            class="pointer-events-none absolute right-3 -bottom-10 left-3 mx-auto w-fit max-w-[calc(100%-1.5rem)] truncate rounded-md bg-black/60 px-3 py-1.5 text-center text-sm text-white shadow-lg"
            title={item.file_name}
        >
            {item.file_name}
        </p>
    {/if}
</Dialog>
