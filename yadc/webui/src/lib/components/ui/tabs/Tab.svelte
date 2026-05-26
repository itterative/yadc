<script lang="ts">
	import { untrack, type Snippet } from 'svelte';
	import { getTabsContext } from './TabsContext.svelte';

	interface Props {
		id: string;
		label: string;
		class?: string;
		children: Snippet;
	}

	let { id, label, class: className = '', children }: Props = $props();

	const ctx = getTabsContext();
	const index = untrack(() => ctx.registerTab(id, label));
	let isActive = $derived(ctx.activeIndex === index);
</script>

<div class={isActive ? className : 'hidden'}>
	{@render children()}
</div>
