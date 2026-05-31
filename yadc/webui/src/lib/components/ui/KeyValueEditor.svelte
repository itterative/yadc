<script lang="ts">
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';

    export type KeyValueType = 'string' | 'number' | 'boolean' | 'object';

    export interface KeyValueEntry {
        key: string;
        value: string | number | boolean | Record<string, unknown>;
        type: KeyValueType;
    }

    interface Props {
        /** The key-value pairs to edit. */
        entries?: KeyValueEntry[];
        /** Placeholder text for key inputs. */
        keyPlaceholder?: string;
        /** Whether the editor is disabled. */
        disabled?: boolean;
        /** HTML id prefix for form elements. */
        idPrefix?: string;
    }

    let {
        entries = $bindable([]),
        keyPlaceholder = 'key',
        disabled = false,
        idPrefix = 'kv'
    }: Props = $props();

    function addEntry() {
        entries = [...entries, { key: '', value: '', type: 'string' }];
    }

    function removeEntry(index: number) {
        entries = entries.filter((_, i) => i !== index);
    }

    function updateKey(index: number, newKey: string) {
        entries = entries.map((e, i) => (i === index ? { ...e, key: newKey } : e));
    }

    function updateType(index: number, newType: KeyValueType) {
        const entry = entries[index];
        let newValue: string | number | boolean | Record<string, unknown>;

        if (newType === 'boolean') {
            newValue = entry.type === 'boolean' ? (entry.value as boolean) : true;
        } else if (newType === 'number') {
            if (entry.type === 'number') {
                newValue = entry.value as number;
            } else if (entry.type === 'boolean') {
                newValue = entry.value ? 1 : 0;
            } else {
                const parsed = parseFloat(String(entry.value));
                newValue = isNaN(parsed) ? 0 : parsed;
            }
        } else if (newType === 'string') {
            newValue = String(entry.value);
        } else {
            // object — keep existing value or empty object for new
            newValue = entry.type === 'object' ? entry.value : {};
        }

        entries = entries.map((e, i) =>
            i === index ? { ...e, type: newType, value: newValue } : e
        );
    }

    function updateStringValue(index: number, newValue: string) {
        entries = entries.map((e, i) => (i === index ? { ...e, value: newValue } : e));
    }

    function updateNumberValue(index: number, newValue: string) {
        const parsed = parseFloat(newValue);
        entries = entries.map((e, i) =>
            i === index ? { ...e, value: isNaN(parsed) ? 0 : parsed } : e
        );
    }

    function updateBooleanValue(index: number, newValue: boolean) {
        entries = entries.map((e, i) => (i === index ? { ...e, value: newValue } : e));
    }

    function isValidNumber(value: string | number | boolean | Record<string, unknown>): boolean {
        if (typeof value === 'number') {
            return true;
        }
        const parsed = parseFloat(String(value));
        return !isNaN(parsed) && String(value).trim() !== '';
    }

    const typeLabels: Record<KeyValueType, string> = {
        string: 'str',
        number: 'num',
        boolean: 'bool',
        object: 'obj'
    };
</script>

<div class="space-y-1.5">
    {#each entries as entry, i (i)}
        {@const isObject = entry.type === 'object'}
        {@const isNumber = entry.type === 'number'}
        {@const isBoolean = entry.type === 'boolean'}
        {@const numberInvalid = isNumber && !isValidNumber(entry.value)}
        <div class="flex items-center gap-1.5">
            <!-- Key input -->
            <input
                id="{idPrefix}-key-{i}"
                type="text"
                class="input min-w-0 flex-1 py-1 text-sm"
                value={entry.key}
                placeholder={keyPlaceholder}
                {disabled}
                oninput={(e) => updateKey(i, e.currentTarget.value)}
            />
            <!-- Type selector -->
            <select
                id="{idPrefix}-type-{i}"
                class="input w-16 shrink-0 cursor-pointer py-1 pr-6 text-[11px]"
                value={entry.type}
                {disabled}
                onchange={(e) => updateType(i, e.currentTarget.value as KeyValueType)}
            >
                {#each Object.entries(typeLabels) as [type, label] (type)}
                    <option value={type}>{label}</option>
                {/each}
            </select>
            <span class="text-xs text-gray-500">=</span>
            <!-- Value input -->
            {#if isBoolean}
                <label class="flex min-w-0 flex-1 cursor-pointer items-center gap-2 py-1">
                    <input
                        id="{idPrefix}-val-{i}"
                        type="checkbox"
                        class="h-4 w-4 accent-accent"
                        checked={Boolean(entry.value)}
                        {disabled}
                        onchange={(e) => updateBooleanValue(i, e.currentTarget.checked)}
                    />
                    <span class="text-xs text-gray-400">{entry.value ? 'true' : 'false'}</span>
                </label>
            {:else if isObject}
                <div
                    class="input min-w-0 flex-1 truncate overflow-hidden bg-gray-900/50 py-1 text-xs text-gray-400"
                    title="Edit in Advanced view"
                >
                    {JSON.stringify(entry.value)}
                </div>
            {:else}
                <input
                    id="{idPrefix}-val-{i}"
                    type={isNumber ? 'number' : 'text'}
                    class="input min-w-0 flex-1 py-1 text-sm {numberInvalid
                        ? 'border-error text-error'
                        : ''}"
                    value={isNumber ? String(entry.value) : (entry.value as string)}
                    placeholder={isNumber ? '0' : 'value'}
                    {disabled}
                    oninput={(e) =>
                        isNumber
                            ? updateNumberValue(i, e.currentTarget.value)
                            : updateStringValue(i, e.currentTarget.value)}
                />
            {/if}
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
        {#if numberInvalid}
            <p class="text-[11px] text-error">Invalid number</p>
        {/if}
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
