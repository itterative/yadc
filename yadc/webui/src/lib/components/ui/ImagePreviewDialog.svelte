<script lang="ts">
    // Generalized full-image lightbox. Callers pass a ready ``src``
    // (a URL or data URL) — they know how to resolve their image.
    // Optional ``caption`` (e.g. filename) and prev/next nav (gallery
    // use). Domain-agnostic so it can be shared by the dataset browser
    // (server images via ``mediaUrl``) and the prompt-example editor
    // (base64 data URLs).
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import SvgChevronLeft from '$lib/icons/SvgChevronLeft.svelte';
    import SvgChevronRight from '$lib/icons/SvgChevronRight.svelte';

    interface Props {
        open: boolean;
        onclose: () => void;
        /** Image source — a URL or data URL. */
        src: string;
        alt?: string;
        /** CSS aspect-ratio value (e.g. ``"1920 / 1080"``). When given,
         *  reserves the box before the image decodes (avoids layout
         *  shift). Omit to size from intrinsic dimensions. */
        aspectRatio?: string;
        width?: number;
        height?: number;
        /** Optional caption shown at the bottom (e.g. filename). */
        caption?: string;
        /** Optional gallery nav. Both handlers must be set for the
         *  arrows to render. */
        onprev?: () => void;
        onnext?: () => void;
        canprev?: boolean;
        cannext?: boolean;
        isloadingnext?: boolean;
    }

    let {
        open,
        onclose,
        src,
        alt = '',
        aspectRatio = undefined,
        width = undefined,
        height = undefined,
        caption = undefined,
        onprev,
        onnext,
        canprev = true,
        cannext = true,
        isloadingnext = false
    }: Props = $props();

    let showArrows = $derived(onprev !== undefined && onnext !== undefined);

    // Reserve horizontal space for the nav arrows (positioned at
    // -left-16 / -right-16) only when they're shown; otherwise let the
    // dialog use the full 90vw.
    let dialogClass = $derived(
        showArrows
            ? 'relative m-auto flex max-h-[calc(90dvh-4rem)] max-w-[calc(90vw-12rem)] flex-col items-center'
            : 'relative m-auto flex max-h-[calc(90dvh-4rem)] max-w-[90vw] flex-col items-center'
    );

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

<Dialog class={dialogClass} {open} {onclose}>
    {#if src}
        <img
            {src}
            {alt}
            class="max-h-[inherit] w-full flex-1 rounded-xl object-contain shadow-2xl"
            style:aspect-ratio={aspectRatio}
            {width}
            {height}
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

        {#if caption}
            <p
                class="pointer-events-none absolute right-3 -bottom-10 left-3 mx-auto w-fit max-w-[calc(100%-1.5rem)] truncate rounded-md bg-black/60 px-3 py-1.5 text-center text-sm text-white shadow-lg"
                title={caption}
            >
                {caption}
            </p>
        {/if}
    {/if}
</Dialog>
