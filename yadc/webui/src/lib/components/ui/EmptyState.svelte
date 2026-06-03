<script lang="ts">
    import type { Snippet } from 'svelte';

    interface Props {
        /** Main heading text. */
        title: string;
        /** Optional secondary text below the title. */
        description?: string;
        /** Optional content rendered below the description (e.g. action buttons). */
        actions?: Snippet;
        /** Optional additional classes for the outer div (e.g. for tighter padding). */
        class?: string;
    }

    let { title, description, actions, class: className = '' }: Props = $props();
</script>

<div class="empty-state {className}">
    <h2 class="text-lg">{title}</h2>
    {#if description}
        <p class="mt-1 text-sm">{description}</p>
    {/if}
    {#if actions}
        {@render actions()}
    {/if}
</div>
