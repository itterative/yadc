<script lang="ts">
    import type { Snippet } from 'svelte';
    import SvgMenuLeft from '$lib/icons/SvgMenuLeft.svelte';

    interface Props {
        /**
         * Whether the panel is open. Bindable so the caller can drive
         * the open/closed state from outside (e.g. open on item click).
         * On desktop (`lg`), the panel is always visible regardless of
         * this value — the binding only affects the mobile drawer.
         */
        open?: boolean;
        /**
         * Extra Tailwind classes merged onto the panel container.
         * Intended primarily for width overrides (e.g. `lg:w-96`),
         * but any visual aspect of the panel container can be tweaked.
         */
        class?: string;
        /**
         * Fired when the panel is closed via an outside click.
         * The X-button inside `children` is the caller's responsibility
         * and should mutate `open` (via `bind:open`) directly.
         */
        onclose?: () => void;
        /** Panel content. Typically a `PillTabs` host with tab children. */
        children: Snippet;
    }

    let { open = $bindable(false), class: className = '', onclose, children }: Props = $props();

    let panelRef: HTMLElement | undefined = $state();

    function handleWindowClick(e: MouseEvent) {
        if (!open || !panelRef) {
            return;
        }
        // Use ``composedPath()`` rather than ``panelRef.contains(target)``.
        // Svelte 5 flushes state changes synchronously after the click
        // handler returns, which can detach (move out of the DOM tree)
        // elements that the user just clicked — by the time this listener
        // runs during the bubble phase, the click target may already be
        // outside the panel's tree. ``composedPath()`` captures the path
        // at dispatch time, before any re-renders, so it still reflects
        // the DOM as it was when the user clicked.
        if (e.composedPath().includes(panelRef)) {
            return;
        }
        open = false;
        onclose?.();
    }

    function toggle(e: MouseEvent) {
        // Stop the click from bubbling to ``<svelte:window>`` —
        // otherwise the opening click would immediately satisfy
        // the "outside click" condition above and close the panel
        // it just opened.
        e.stopPropagation();
        open = !open;
    }
</script>

<svelte:window onclick={handleWindowClick} />

<!-- Visual dim backdrop. No click handler — closing on outside
     click is handled by the ``<svelte:window>`` listener above so
     the click doesn't have to land on the dimmed region specifically.
     Marked ``role="presentation"`` so AT ignores it (it's purely
     decorative). Hidden at ``lg+`` where the panel sits inline. -->
{#if open}
    <div
        class="fixed inset-0 z-30 bg-black/50 transition-opacity lg:hidden"
        role="presentation"
    ></div>
{/if}

<!-- Panel container. Mobile: fixed drawer sliding from the right.
     Desktop (``lg+``): static, inline, always visible. The closing
     ``translate-x-full`` is overridden by ``lg:translate-x-0`` so
     the desktop panel never slides off-screen. -->
<div
    class="side-panel
    fixed inset-y-0 right-0 z-40 w-[80vw] max-w-120 shadow-2xl transition-transform duration-300
    lg:static lg:w-120 lg:max-w-none lg:shrink-0 lg:shadow-none lg:transition-none
    {open ? '' : 'translate-x-full'} lg:translate-x-0
    {className}"
    bind:this={panelRef}
>
    <!-- Inner chrome — visual styling only. Callers put their content
         (typically a tab host) directly inside. ``h-full flex-col`` so
         the tab host can fill the available height with ``flex-1``. -->
    <div
        class="flex h-full flex-col overflow-hidden border-l border-border bg-surface lg:rounded-xl lg:border"
    >
        {@render children()}
    </div>
</div>

<!-- Mobile-only floating toggle button. Always rendered on mobile so
     the user can re-open the drawer after closing it; the backdrop
     (``z-30``) and panel (``z-40``) sit above it, so it's hidden
     while the drawer is open. -->
<button
    class="fixed right-6 bottom-6 z-20 flex h-16 w-16 cursor-pointer items-center justify-center rounded-full
         bg-accent text-white shadow-lg transition-colors hover:bg-accent-hover lg:hidden"
    onclick={toggle}
    title="Toggle panel"
    aria-label="Toggle panel"
>
    <SvgMenuLeft class="h-6 w-6" />
</button>
