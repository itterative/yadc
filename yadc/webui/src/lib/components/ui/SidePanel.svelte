<script lang="ts">
    import type { Snippet } from 'svelte';
    import FabButton from './FabButton.svelte';
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
        /**
         * Optional override for the mobile FAB. Render a
         * ``<FabButton>`` inside — positioning (``fixed right-6
         * bottom-6 z-20 lg:hidden``) and click-containment are
         * owned by the wrapper here, so the snippet only chooses
         * *what* the button is (icon/variant/label/action), not its
         * shape. Use it to swap the FAB for a context action (e.g.
         * a streaming cancel). The default (no snippet) is the
         * panel toggle.
         */
        fab?: Snippet;
        /** Panel content. Typically a `PillTabs` host with tab children. */
        children: Snippet;
    }

    let {
        open = $bindable(false),
        class: className = '',
        onclose,
        children,
        fab
    }: Props = $props();

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

<!-- Mobile-only floating action button. Always rendered on mobile so
     the user can interact while the panel is closed; the backdrop
     (``z-30``) and panel (``z-40``) sit above it, so it's hidden
     while the drawer is open. The wrapper owns positioning AND
     click-containment — its ``onclick`` stops propagation so a FAB
     click never bubbles to ``<svelte:window>`` and re-closes the
     panel via the outside-click listener. That stopPropagation is
     load-bearing for the default toggle (it opens the panel on the
     very click that would otherwise re-close it). Default content
     is the toggle; provide the ``fab`` snippet to swap in a context
     action (typically a ``<FabButton>``). The wrapper's ``onclick``
     is a propagation guard, not an interaction — the actual control
     is the ``<button>`` inside ``FabButton`` — so the a11y rules for
     interactive elements don't apply. -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="fixed right-6 bottom-6 z-20 lg:hidden" onclick={(e) => e.stopPropagation()}>
    {#if fab}
        {@render fab()}
    {:else}
        <FabButton icon={SvgMenuLeft} label="Toggle panel" onclick={() => (open = !open)} />
    {/if}
</div>
