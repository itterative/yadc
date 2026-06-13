<script lang="ts">
    import type { Snippet } from 'svelte';

    interface Props {
        class?: string;
        open: boolean;
        onclose: () => void;
        children: Snippet;
    }

    let dialog: HTMLDialogElement | null = $state(null);
    let container: HTMLElement | null = $state(null);

    let { class: klazz = '', open, onclose, children }: Props = $props();

    // True when the parent is closing the dialog via the `open` prop. Prevents
    // the native `close` event from re-invoking `onclose()` after the parent
    // already initiated the close (e.g. child close button calls onclose()).
    let closingFromParent = $state(false);

    $effect(() => {
        if (dialog === null) {
            return;
        }

        if (open) {
            closingFromParent = false;
            if (!dialog.open) {
                dialog.showModal();
                document.body.classList.add('noscroll');
            }
        } else {
            if (dialog.open) {
                closingFromParent = true;
                dialog.close();
            }
            document.body.classList.remove('noscroll');
        }
    });

    function handleClose() {
        document.body.classList.remove('noscroll');
        if (closingFromParent) {
            closingFromParent = false;
            return;
        }
        onclose();
    }

    function handleDialogClick(ev: MouseEvent) {
        if (container == null || ev.target == null) {
            return;
        }

        if (container.contains(ev.target as Node)) {
            return;
        }

        // closedby="any" already closed the dialog and fired the close event;
        // avoid calling close() again (and don't fire onclose twice).
        if (!dialog?.open) {
            return;
        }

        dialog.close();
    }
</script>

<dialog
    class="fixed top-0 right-0 bottom-0 left-0 z-50 hidden h-full max-h-none w-full max-w-none justify-center overflow-y-hidden overscroll-contain bg-transparent p-4 backdrop:bg-black backdrop:opacity-80 open:flex"
    bind:this={dialog}
    onclose={handleClose}
    onclick={handleDialogClick}
    closedby="any"
>
    <div class={klazz} bind:this={container}>
        {@render children()}
    </div>
</dialog>
