<script lang="ts">
    import type { Component } from 'svelte';

    interface Props {
        /** Icon component (an ``Svg*``). Sized by FabButton to keep the FAB's visual language uniform. */
        icon: Component<{ class?: string }>;
        /** Accessible label + tooltip. The FAB is icon-only, so this is the only text AT announces. */
        label: string;
        /** Click handler. */
        onclick: () => void;
        /** Color. */
        variant?: 'accent' | 'error';
    }

    let { icon: Icon, label, onclick, variant = 'accent' }: Props = $props();

    const variantClasses = {
        accent: 'bg-accent hover:bg-accent-hover',
        error: 'bg-error hover:bg-error/80'
    };
</script>

<button
    type="button"
    class="flex h-16 w-16 cursor-pointer items-center justify-center rounded-full text-white shadow-lg transition-colors {variantClasses[
        variant
    ]}"
    {onclick}
    title={label}
    aria-label={label}
>
    <Icon class="h-6 w-6" />
</button>
