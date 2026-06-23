<script lang="ts">
    import SvgSparkle from '$lib/icons/SvgSparkle.svelte';

    interface Props {
        /** Master toggle — when false, nothing renders. */
        active?: boolean;
        /** Container classes (color via `currentColor`, e.g. `text-accent`). */
        class?: string;
    }

    let { active = false, class: klazz = '' }: Props = $props();

    // Fixed trio — big bottom-left pulses independently, medium + small
    // drift as a constellation. ``left`` and ``bottom`` are the base
    // position. ``translateX(-50%)`` centers each sparkle on its ``left``
    // (the keyframes bake it in; the inline base mirrors it so the
    // reduced-motion fallback stays centered).
    const sparkles = [
        { style: 'left: 50%; bottom: 0px; width: 2rem; height: 2rem', delay: '0s' },
        { style: 'left: 75%; bottom: 35%; width: 1rem; height: 1rem', delay: '0.6s' },
        { style: 'left: 62%; bottom: 55.5%; width: 0.75rem; height: 0.75rem', delay: '1.2s' }
    ];
</script>

<span class="relative inline-block h-12 w-12 align-middle {klazz}" aria-hidden="true">
    {#each sparkles as s, i (i)}
        <span
            class="absolute"
            class:sparkle-think={active && i > 0}
            class:sparkle-pulse={active && i === 0}
            style="{s.style}; transform:translateX(-50%); animation-delay:{s.delay}"
        >
            <SvgSparkle class="h-full w-full" />
        </span>
    {/each}
</span>
