<script lang="ts" module>
    /** A single overridden field with its display label. The `field` value is opaque
     *  to this section — it's passed back through `onReset` for the parent to handle. */
    export interface OverrideItem {
        field: string;
        label: string;
    }
</script>

<script lang="ts">
    import SvgChevronLeft from '$lib/icons/SvgChevronLeft.svelte';

    interface Props {
        /** List of overridden fields with their labels. The section renders one row
         *  per item with a "Reset" button. */
        overrides: ReadonlyArray<OverrideItem>;
        /** Whether the section is expanded. Bindable. */
        expanded?: boolean;
        /** Called when the user clicks "Reset" on a single field. */
        onReset: (field: string) => void;
        /** Called when the user clicks "Reset all overrides". */
        onResetAll: () => void;
    }

    let { overrides, expanded = $bindable(false), onReset, onResetAll }: Props = $props();
</script>

{#if overrides.length > 0}
    <section class="overflow-hidden rounded-lg border border-border">
        <button
            class="flex w-full cursor-pointer items-center gap-2 px-3 py-2.5 text-sm text-gray-300 transition-colors hover:bg-surface"
            onclick={() => (expanded = !expanded)}
        >
            <SvgChevronLeft
                class="h-4 w-4 transition-transform {expanded ? '-rotate-90' : '-rotate-180'}"
            />
            <span class="flex-1 text-left">Overrides</span>
            <span class="badge-accent">{overrides.length}</span>
        </button>

        {#if expanded}
            <div class="space-y-1 border-t border-border px-3 py-2">
                {#each overrides as item (item.field)}
                    <div class="flex items-center justify-between py-1.5">
                        <span class="text-sm text-gray-300">{item.label}</span>
                        <button
                            class="cursor-pointer text-xs text-accent transition-colors hover:text-accent-hover"
                            onclick={() => onReset(item.field)}
                        >
                            Reset
                        </button>
                    </div>
                {/each}

                <div class="border-t border-border pt-1">
                    <button
                        class="w-full cursor-pointer py-1.5 text-xs text-gray-400 transition-colors hover:text-white"
                        onclick={onResetAll}
                    >
                        Reset all overrides
                    </button>
                </div>
            </div>
        {/if}
    </section>
{/if}
