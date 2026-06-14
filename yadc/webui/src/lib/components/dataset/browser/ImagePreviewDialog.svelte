<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import SvgChevronLeft from '$lib/icons/SvgChevronLeft.svelte';
    import SvgChevronRight from '$lib/icons/SvgChevronRight.svelte';
    import { mediaUrl, type ImageInfo } from '$lib/stores/dataset';

    interface Props {
        open: boolean;
        onclose: () => void;
        onprev?: () => void;
        onnext?: () => void;
        canprev?: boolean;
        cannext?: boolean;
        isloadingnext?: boolean;
        datasetName: string;
        item: ImageInfo | null;
    }

    let {
        open,
        onclose,
        onprev,
        onnext,
        canprev = true,
        cannext = true,
        isloadingnext = false,
        datasetName,
        item
    }: Props = $props();

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

    let showArrows = $derived(onprev !== undefined && onnext !== undefined);

    function handleKeydown(ev: KeyboardEvent) {
        if (!open) {
            return;
        }
        if (ev.key === 'ArrowLeft' && onprev && canprev) {
            ev.preventDefault();
            onprev();
        } else if (ev.key === 'ArrowRight' && onnext && cannext && !isloadingnext) {
            ev.preventDefault();
            onnext();
        }
    }
</script>

<svelte:window onkeydown={handleKeydown} />

<Dialog
    class="relative m-auto flex max-h-[calc(90dvh-4rem)] max-w-[calc(90vw-12rem)] flex-col items-center"
    {open}
    {onclose}
>
    {#if item !== null}
        <img
            src={imgSrc}
            alt={item.file_name}
            class="max-h-[inherit] w-full flex-1 rounded-xl object-contain shadow-2xl"
            style:aspect-ratio={aspectRatio}
            width={item.width || undefined}
            height={item.height || undefined}
        />

        {#if showArrows}
            <button
                class="absolute top-1/2 -left-16 -translate-y-1/2 cursor-pointer rounded-full bg-black/50 p-2.5 text-white shadow-lg transition-colors hover:bg-black/70 disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-black/50"
                onclick={onprev}
                disabled={!canprev}
                aria-label="Previous image"
                title="Previous image (←)"
            >
                <SvgChevronLeft class="h-8 w-8" />
            </button>
            <button
                class="absolute top-1/2 -right-16 -translate-y-1/2 cursor-pointer rounded-full bg-black/50 p-2.5 text-white shadow-lg transition-colors hover:bg-black/70 disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-black/50"
                onclick={onnext}
                disabled={!cannext || isloadingnext}
                aria-label="Next image"
                title={isloadingnext ? 'Loading more images…' : 'Next image (→)'}
            >
                <SvgChevronRight class="h-8 w-8" />
            </button>
        {/if}

        <p
            class="pointer-events-none absolute right-3 -bottom-10 left-3 mx-auto w-fit max-w-[calc(100%-1.5rem)] truncate rounded-md bg-black/60 px-3 py-1.5 text-center text-sm text-white shadow-lg"
            title={item.file_name}
        >
            {item.file_name}
        </p>
    {/if}
</Dialog>
