<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    class?: string;
    open: boolean;
    onclose: () => void;
    children: Snippet;
  }

  let dialog: HTMLDialogElement | null = $state(null);
  let container: HTMLElement | null = $state(null);

  let { class: klazz = "", open, onclose, children }: Props = $props();

  // Track open state internally for smooth transitions
  let dialogOpen = $state(false);

  $effect(() => {
    dialogOpen = open;
  });

  $effect(() => {
    if (dialog === null) {
      return;
    }

    if (dialogOpen && !dialog.open) {
      dialog.showModal();
      document.body.classList.add("noscroll");
    } else if (!dialogOpen && dialog.open) {
      dialog.close();
      document.body.classList.remove("noscroll");
    }
  });

  function handleDialogClick(ev: MouseEvent) {
    if (container == null || ev.target == null) {
      return;
    }

    if (container.contains(ev.target as Node)) {
      return;
    }

    closeDialog();
  }

  function closeDialog(ev?: Event) {
    if (ev) {
      ev.preventDefault();
    }

    dialogOpen = false;
    onclose();
  }
</script>

<dialog
  class="fixed top-0 bottom-0 left-0 right-0 hidden open:flex w-full max-w-none h-full max-h-none justify-center bg-transparent overscroll-contain z-50 overflow-y-hidden backdrop:bg-black backdrop:opacity-80 p-4"
  bind:this={dialog}
  onclose={onclose}
  onclick={handleDialogClick}
  closedby="any"
>
  <div class={klazz} bind:this={container}>
    {@render children()}
  </div>
</dialog>
