<script lang="ts">
    interface Props {
        id: string;
        /** Controlled value (one-way); commit changes via ``oninput``. */
        value: number;
        min?: number;
        max?: number;
        step?: number;
        disabled?: boolean;
        /** Accessible name — the visible ``<label>`` may sit on a sibling control. */
        ariaLabel?: string;
        /** Fired with the new numeric value while dragging / arrowing. */
        oninput?: (value: number) => void;
        /** Fired with the committed value on release / change. */
        onchange?: (value: number) => void;
    }

    let {
        id,
        value,
        min = 0,
        max = 1,
        step = 0.01,
        disabled = false,
        ariaLabel,
        oninput,
        onchange
    }: Props = $props();

    let el: HTMLInputElement | undefined = $state();

    // Native range inputs only honor the ``value`` *attribute* at creation; a
    // later property change is what actually moves the thumb. Svelte's reactive
    // ``value`` binding can miss the first paint (the thumb then rests at a
    // browser default), so we force the property on mount and every change.
    $effect(() => {
        if (el) {
            el.value = String(value);
        }
    });
</script>

<!-- Native range input: native keyboard/touch/screen-reader support for free.
     accent-color themes both the thumb and the filled track in modern browsers. -->
<input
    bind:this={el}
    type="range"
    {id}
    {value}
    {min}
    {max}
    {step}
    {disabled}
    aria-label={ariaLabel}
    oninput={(e) => oninput?.(Number((e.currentTarget as HTMLInputElement).value))}
    onchange={(e) => onchange?.(Number((e.currentTarget as HTMLInputElement).value))}
    class="w-full cursor-pointer accent-accent disabled:cursor-not-allowed disabled:opacity-50"
/>
