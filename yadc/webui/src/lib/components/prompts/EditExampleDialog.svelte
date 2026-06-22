<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import ImagePreviewDialog from '$lib/components/ui/ImagePreviewDialog.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgFullscreen from '$lib/icons/SvgFullscreen.svelte';

    interface Props {
        open: boolean;
        /** Dialog heading — "Add example" vs "Edit example". */
        title?: string;
        /** The example's image, shown read-only. Not replaceable here —
         *  to change the image, delete the example and re-add it. */
        imageDataUrl: string;
        initialSubject: string;
        initialCaption: string;
        onsave: (patch: { subject: string; caption: string }) => void;
        onclose: () => void;
    }

    let {
        open,
        title = 'Example',
        imageDataUrl,
        initialSubject,
        initialCaption,
        onsave,
        onclose
    }: Props = $props();

    let subject = $state('');
    let caption = $state('');

    // Full-image lightbox. The hero banner crops (object-cover, like
    // RefineDialog), so the whole image is only visible here.
    let zoomOpen = $state(false);

    // Reset from the seed values each time the dialog opens, so edits
    // don't leak across open/close cycles and a freshly-picked file's
    // seed is picked up. Mirrors EditTemplateDialog's reset-on-open.
    $effect(() => {
        if (!open) {
            return;
        }
        subject = initialSubject;
        caption = initialCaption;
        zoomOpen = false;
    });

    function handleSave() {
        onsave({ subject: subject.trim(), caption });
    }
</script>

<Dialog class="dialog-panel flex max-w-lg flex-col overflow-hidden" {open} {onclose}>
    <div class="flex flex-col p-5">
        <div class="dialog-header">
            <h2 class="dialog-title">{title}</h2>
            <button class="btn-close" onclick={onclose} aria-label="Close">
                <SvgClose class="h-5 w-5" />
            </button>
        </div>

        <!-- Hero banner (RefineDialog pattern): breaks out of the dialog
             padding to span full width, fixed height, object-cover. Crops
             by design — click opens the lightbox below for the full image.
             The fullscreen chip is a non-interactive affordance hint
             (pointer-events-none); the whole banner is the click target. -->
        <button
            type="button"
            class="relative -mx-5 mb-4 box-content block h-32 w-[calc(100%+var(--spacing)*10)] max-w-none cursor-zoom-in overflow-hidden bg-gray-700 transition-opacity hover:opacity-90"
            onclick={() => (zoomOpen = true)}
            title="View full image"
            aria-label="View full image"
        >
            <img src={imageDataUrl} alt={subject} class="h-full w-full object-cover" />
            <span
                class="pointer-events-none absolute right-2 bottom-2 flex items-center justify-center rounded-md bg-black/60 p-1.5 text-gray-300"
            >
                <SvgFullscreen class="h-4 w-4" />
            </span>
        </button>

        <div class="space-y-3">
            <div>
                <label class="label" for="example-subject">Subject</label>
                <input
                    id="example-subject"
                    type="text"
                    bind:value={subject}
                    class="input"
                    placeholder="Short label for this image"
                />
            </div>
            <div>
                <label class="label" for="example-caption">Caption</label>
                <textarea
                    id="example-caption"
                    bind:value={caption}
                    class="input h-28 resize-y"
                    placeholder="Caption text the model should learn to produce"
                ></textarea>
            </div>
        </div>

        <div class="btn-bar mt-4">
            <button class="btn-secondary" onclick={onclose}>Cancel</button>
            <button class="btn-primary" onclick={handleSave}>Save</button>
        </div>
    </div>
</Dialog>

<!-- Full-image lightbox (shared ``ImagePreviewDialog`` — same one
     the dataset browser gallery uses). The hero banner above crops
     (object-cover), so the whole image is only visible here. -->
<ImagePreviewDialog
    open={zoomOpen}
    onclose={() => (zoomOpen = false)}
    src={imageDataUrl}
    alt={subject}
/>
