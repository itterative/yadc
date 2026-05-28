<script lang="ts">
    import type { Snippet } from 'svelte';

    type Direction = 'top' | 'right' | 'bottom' | 'left';

    interface Props {
        class?: string;
        label: string;
        direction: Direction;
        children: Snippet;
    }

    let { class: klazz = '', label, direction, children }: Props = $props();

    const positionClasses: Record<Direction, string> = {
        top: 'bottom-full left-1/2 mb-2 -translate-x-1/2',
        right: 'top-1/2 left-full ml-2 -translate-y-1/2',
        bottom: 'top-full left-1/2 mt-2 -translate-x-1/2',
        left: 'top-1/2 right-full mr-2 -translate-y-1/2'
    };
</script>

<div class="group relative flex {klazz}">
    {@render children()}
    <span
        class="pointer-events-none absolute rounded bg-fg px-2 py-1 text-xs whitespace-nowrap text-bg opacity-0 transition-opacity duration-150 group-hover:opacity-100 max-md:hidden {positionClasses[
            direction
        ]}"
        role="tooltip"
    >
        {label}
    </span>
</div>
