<script lang="ts">
    import { untrack, type Snippet, type Component } from 'svelte';
    import { getTabsContext } from './TabsContext.svelte';

    interface Props {
        id: string;
        label: string;
        icon?: Component<{ class?: string }>;
        class?: string;
        children: Snippet;
    }

    let { id, label, icon, class: className = '', children }: Props = $props();

    const ctx = getTabsContext();
    const index = untrack(() => ctx.registerTab(id, label, icon));
    let isActive = $derived(ctx.activeIndex === index);
</script>

<div class={isActive ? `${className} min-h-0 flex-1` : 'hidden'}>
    {@render children()}
</div>
