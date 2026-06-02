<script lang="ts">
  import { toasts } from "$lib/stores/toasts";
  import ToastItem from "./ToastItem.svelte";
  import { fly, fade } from "svelte/transition";
</script>

{#if $toasts.length > 0}
  <div class="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
    {#each $toasts as t (t.id)}
      <div
        class="pointer-events-auto"
        in:fly={{ duration: 200, y: 20, opacity: 0.8 }}
        out:fade={{ duration: 150 }}
      >
        <ToastItem toast={t} />
      </div>
    {/each}
  </div>
{/if}
