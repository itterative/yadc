<script lang="ts">
    import { type Snippet } from 'svelte';

    export interface MenuItem {
        label: string;
        onClick: () => void;
    }

    interface Props {
        children: Snippet;
        items: MenuItem[];
        class?: string;
    }

    let { children, items, class: klazz = '' }: Props = $props();

    let x = $state(0);
    let y = $state(0);
    let menuEl: HTMLDivElement | null = $state(null);
    let triggerEl: HTMLSpanElement | null = $state(null);
    let scrollCleanups: (() => void)[] = [];

    function isScrollable(el: Element): boolean {
        const style = getComputedStyle(el);
        return (
            style.overflow === 'auto' ||
            style.overflow === 'scroll' ||
            style.overflowY === 'auto' ||
            style.overflowY === 'scroll'
        );
    }

    function getScrollableAncestors(el: Element | null): Element[] {
        const ancestors: Element[] = [];
        let current = el?.parentElement ?? null;
        while (current) {
            if (isScrollable(current)) {
                ancestors.push(current);
            }
            current = current.parentElement;
        }
        return ancestors;
    }

    function closeMenu() {
        if (menuEl) {
            menuEl.hidePopover();
        }
    }

    function attachScrollListeners() {
        detachScrollListeners();
        const ancestors = getScrollableAncestors(triggerEl);
        for (const ancestor of ancestors) {
            const handler = () => {
                closeMenu();
            };
            ancestor.addEventListener('scroll', handler, { passive: true });
            scrollCleanups.push(() => {
                ancestor.removeEventListener('scroll', handler);
            });
        }
    }

    function detachScrollListeners() {
        for (const cleanup of scrollCleanups) {
            cleanup();
        }
        scrollCleanups = [];
    }

    async function handleContextMenu(e: MouseEvent) {
        if (!menuEl || !triggerEl) {
            return;
        }

        e.preventDefault();
        x = e.clientX;
        y = e.clientY;

        menuEl.showPopover();

        const rect = menuEl.getBoundingClientRect();
        const triggerRect = triggerEl.getBoundingClientRect();

        if (x < triggerRect.left + 4) {
            x = triggerRect.left + 4;
        }

        if (x + rect.width > window.innerWidth) {
            x = triggerRect.left + triggerRect.width - rect.width;
        }

        if (y < triggerRect.top + 4) {
            y = triggerRect.top + 4;
        }

        if (y + rect.height > window.innerHeight) {
            y = triggerRect.top - rect.height - 4;
        }

        attachScrollListeners();
    }

    function handleWindowClick(e: MouseEvent) {
        if (menuEl && !menuEl.contains(e.target as Node)) {
            menuEl.hidePopover();
        }
    }

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === 'Escape' && menuEl) {
            menuEl.hidePopover();
        }
    }

    function handleBeforeToggle(e: ToggleEvent) {
        if (e.newState === 'closed') {
            detachScrollListeners();
        }
    }

    function handleItemClick(item: MenuItem) {
        item.onClick();
        if (menuEl) {
            menuEl.hidePopover();
        }
    }
</script>

<svelte:window onclick={handleWindowClick} onkeydown={handleKeydown} />

<span bind:this={triggerEl} oncontextmenu={handleContextMenu} class={klazz} role="none">
    {@render children()}
</span>

<div
    bind:this={menuEl}
    popover="manual"
    class="min-w-[10rem] overflow-hidden rounded-lg border border-border bg-surface py-1 shadow-xl"
    style="position: fixed; left: {x}px; top: {y}px; margin: 0;"
    role="menu"
    onbeforetoggle={handleBeforeToggle}
>
    {#each items as item (item.label)}
        <button
            type="button"
            class="w-full cursor-pointer px-3 py-1.5 text-left text-sm text-gray-200 transition-colors hover:bg-gray-700"
            onclick={() => handleItemClick(item)}
            role="menuitem"
        >
            {item.label}
        </button>
    {/each}
</div>
