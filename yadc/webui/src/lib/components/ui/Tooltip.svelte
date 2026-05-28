<script lang="ts">
    import type { Snippet } from 'svelte';

    interface Props {
        label: string;
        children: Snippet;
    }

    let { label, children }: Props = $props();
</script>

<div class="group relative flex">
    {@render children()}
    <span
        class="pointer-events-none absolute top-1/2 left-full ml-2 -translate-y-1/2 rounded bg-fg px-2 py-1 text-xs whitespace-nowrap text-bg opacity-0 transition-opacity duration-150 group-hover:opacity-100 max-md:hidden"
        role="tooltip"
    >
        {label}
    </span>
</div>
