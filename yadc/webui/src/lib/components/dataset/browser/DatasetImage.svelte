<script lang="ts">
    import type { ImageInfo } from '$lib/stores/datasetImages';
    import SvgWarning from '$lib/icons/SvgWarning.svelte';

    interface Props {
        class?: string;
        datasetName: string;
        item: ImageInfo;
        selected?: boolean;
        captioning?: boolean;
        onclick: (item: ImageInfo) => void;
    }

    let {
        class: klazz = '',
        datasetName,
        item,
        selected = false,
        captioning = false,
        onclick
    }: Props = $props();

    let media: HTMLImageElement | null = $state(null);
    let loading: boolean = $state(true);
    let flashing: boolean = $state(false);

    let thumbnailSrc = $derived(
        `/api/datasets/${encodeURIComponent(datasetName)}/images/${item.id}/thumbnail?size=512`
    );

    let aspectRatio = $derived.by(() => {
        if (!item.width || !item.height) {
            return undefined;
        }
        return `${item.width} / ${item.height}`;
    });

    $effect(() => {
        const setLoadingToFalse = () => (loading = false);
        media?.addEventListener('load', setLoadingToFalse);
        return () => media?.removeEventListener('load', setLoadingToFalse);
    });

    // Trigger a one-shot highlight animation when flash changes.
    $effect(() => {
        if (item.flash) {
            flashing = true;
        }
    });

    function handleFlashEnd() {
        flashing = false;
    }
</script>

<button
    class="{klazz} relative {selected
        ? 'outline-2 outline-offset-[-2px] outline-accent'
        : ''} {captioning ? 'shimmer-accent animate-none' : ''} {flashing && !captioning
        ? 'animate-tile-flash'
        : ''}"
    onclick={() => onclick(item)}
    class:has-caption={item.has_caption}
    class:has-toml={item.has_toml}
    onanimationend={handleFlashEnd}
>
    <img
        src={thumbnailSrc}
        alt={item.file_name}
        loading="lazy"
        class="block w-full bg-gray-500"
        class:animate-pulse={loading}
        width={item.width || undefined}
        height={item.height || undefined}
        style:aspect-ratio={aspectRatio}
        bind:this={media}
    />

    {#if item.has_caption}
        <span
            class="badge-sm pointer-events-none absolute top-2 right-2 flex h-4 w-4 items-center justify-center bg-success text-black"
            title="Has caption">✓</span
        >
    {/if}

    {#if item.has_toml}
        <span
            class="badge-sm pointer-events-none absolute top-2 right-7 flex h-4 w-4 items-center justify-center bg-accent text-black"
            title="Has TOML extras">T</span
        >
    {/if}

    {#if item.draft_names.length > 0}
        <span
            class="badge-sm pointer-events-none absolute top-2 left-2 flex h-4 w-4 items-center justify-center bg-error text-white"
            title="{item.draft_names.length} draft(s)">D</span
        >
    {/if}

    {#if item.caption_error}
        <span
            class="badge-sm pointer-events-none absolute bottom-2 left-2 flex h-4 w-4 items-center justify-center bg-warning text-black"
            title={item.caption_error}
        >
            <SvgWarning class="h-full w-full" />
        </span>
    {/if}
</button>
