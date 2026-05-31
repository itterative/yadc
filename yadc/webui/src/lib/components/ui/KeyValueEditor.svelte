<script lang="ts">
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';

    export interface KeyValueEntry {
        key: string;
        value: string;
    }

    interface Props {
        /** The key-value pairs to edit. */
        entries?: KeyValueEntry[];
        /** Placeholder text for key inputs. */
        keyPlaceholder?: string;
        /** Placeholder text for value inputs. */
        valuePlaceholder?: string;
        /** Whether the editor is disabled. */
        disabled?: boolean;
        /** HTML id prefix for form elements. */
        idPrefix?: string;
    }

    let {
        entries = $bindable([]),
        keyPlaceholder = 'key',
        valuePlaceholder = 'value',
        disabled = false,
        idPrefix = 'kv'
    }: Props = $props();

    function addEntry() {
        entries = [...entries, { key: '', value: '' }];
    }

    function removeEntry(index: number) {
        entries = entries.filter((_, i) => i !== index);
    }

    function updateKey(index: number, newKey: string) {
        entries = entries.map((e, i) => (i === index ? { ...e, key: newKey } : e));
    }

    function updateValue(index: number, newValue: string) {
        entries = entries.map((e, i) => (i === index ? { ...e, value: newValue } : e));
    }
</script>

<div class="space-y-1.5">
    {#each entries as entry, i (i)}
        <div class="flex items-center gap-1.5">
            <input
                id="{idPrefix}-key-{i}"
                type="text"
                class="input min-w-0 flex-1 py-1 text-sm"
                value={entry.key}
                placeholder={keyPlaceholder}
                {disabled}
                oninput={(e) => updateKey(i, e.currentTarget.value)}
            />
            <span class="text-xs text-gray-500">=</span>
            <input
                id="{idPrefix}-val-{i}"
                type="text"
                class="input min-w-0 flex-1 py-1 text-sm"
                value={entry.value}
                placeholder={valuePlaceholder}
                {disabled}
                oninput={(e) => updateValue(i, e.currentTarget.value)}
            />
            <button
                class="cursor-pointer p-0.5 text-gray-500 transition-colors hover:text-error {disabled
                    ? 'pointer-events-none opacity-40'
                    : ''}"
                onclick={() => removeEntry(i)}
                {disabled}
                title="Remove"
                aria-label="Remove variable"
            >
                <SvgClose class="h-3.5 w-3.5" />
            </button>
        </div>
    {/each}
    <button
        class="flex cursor-pointer items-center gap-1 text-xs text-accent transition-colors hover:text-accent-hover {disabled
            ? 'pointer-events-none opacity-40'
            : ''}"
        onclick={addEntry}
        {disabled}
    >
        <SvgPlus class="h-3.5 w-3.5" />
        Add variable
    </button>
</div>
