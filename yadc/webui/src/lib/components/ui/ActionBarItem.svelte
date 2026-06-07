<script lang="ts">
    import type { Component, Snippet } from 'svelte';

    interface Props {
        onclick: () => void;
        disabled?: boolean;
        icon?: Component<{ class?: string }>;
        variant?: 'primary' | 'secondary' | 'danger';
        children: Snippet;
    }

    let {
        onclick,
        disabled = false,
        icon: Icon,
        variant = 'secondary',
        children
    }: Props = $props();

    const variantClasses = {
        primary: 'text-accent hover:text-accent-hover',
        secondary: 'text-gray-400 hover:text-gray-200',
        danger: 'text-error hover:text-error/80'
    };
</script>

<div class="flex min-w-0 flex-1 items-center justify-center">
    <button
        class="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg px-3 py-2 transition-colors hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50 {variantClasses[
            variant
        ]}"
        {onclick}
        {disabled}
    >
        {#if Icon}
            <Icon class="h-4 w-4 shrink-0" />
        {/if}
        <span class="min-w-0 truncate">{@render children()}</span>
    </button>
</div>
