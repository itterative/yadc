<script lang="ts">
    import type { Snippet } from 'svelte';

    interface Props {
        label: string;
        children: Snippet;
    }

    let { label, children }: Props = $props();
</script>

<div class="tooltip-wrapper">
    {@render children()}
    <span class="tooltip-label" role="tooltip">{label}</span>
</div>

<style>
    .tooltip-wrapper {
        position: relative;
        display: flex;
    }

    .tooltip-label {
        position: absolute;
        left: calc(100% + 0.5rem);
        top: 50%;
        transform: translateY(-50%);
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        background: var(--color-fg);
        color: var(--color-bg);
        font-size: 0.75rem;
        white-space: nowrap;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.15s;
        z-index: 50;
    }

    .tooltip-wrapper:hover .tooltip-label {
        opacity: 1;
    }

    @media (max-width: 767px) {
        .tooltip-wrapper:hover .tooltip-label {
            display: none;
        }
    }
</style>
