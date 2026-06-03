<script lang="ts">
    import type { Snippet } from 'svelte';

    interface Props {
        class?: string;
        children: Snippet;
    }

    let { class: className = '', children }: Props = $props();
</script>

<div class="overflow-hidden rounded-lg bg-gray-800 {className}">
    {@render children()}
</div>
