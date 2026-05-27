<script lang="ts">
    import type { Snippet } from 'svelte';
    import { setTopbarContent } from '$lib/stores/topbar.svelte';

    interface Props {
        children: Snippet<[]>;
    }

    let { children }: Props = $props();

    $effect(() => {
        setTopbarContent(children);
        return () => setTopbarContent(null);
    });
</script>
