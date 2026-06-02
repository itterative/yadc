<script lang="ts">
	interface Tab {
		value: string;
		label: string;
	}

	interface Props {
		tabs: Tab[];
		selected: string;
		onchange: (value: string) => void;
		class?: string;
	}

	let { tabs, selected, onchange, class: klazz = '' }: Props = $props();
</script>

<div class="flex gap-1 border-b border-border {klazz}">
	{#each tabs as tab (tab.value)}
		<button
			class="cursor-pointer rounded-t-lg px-4 py-2 text-sm font-medium transition-colors {selected ===
			tab.value
				? 'border-b-2 border-accent bg-surface text-accent'
				: 'text-gray-400 hover:text-gray-200'}"
			onclick={() => onchange(tab.value)}
		>
			{tab.label}
		</button>
	{/each}
</div>
