<script lang="ts">
    import type { Snippet } from 'svelte';

    interface Props {
        children: Snippet;
    }

    let { children }: Props = $props();
</script>

<div class="flex flex-nowrap gap-2 bg-black/15 p-2 text-sm">
    {@render children()}
</div>
