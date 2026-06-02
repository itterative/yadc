<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    open: boolean;
    oncancel: () => void;
    onconfirm: () => void;
    children: Snippet;
  }

  let { open, oncancel, onconfirm, children }: Props = $props();
</script>

{#if open}
  <div class="p-4 rounded-lg bg-error/10 border border-error/20">
    <p class="text-sm text-gray-200 mb-3">{@render children()}</p>
    <div class="btn-bar">
      <button class="btn-secondary px-3 py-1.5" onclick={oncancel}>Cancel</button>
      <button class="btn-danger" onclick={onconfirm}>Delete</button>
    </div>
  </div>
{/if}
