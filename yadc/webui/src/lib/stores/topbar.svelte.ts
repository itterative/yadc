import type { Snippet } from 'svelte';

/**
 * Shared topbar snippet store.
 * Pages set this via <Topbar> to provide contextual header content
 * (title, status, progress) that renders in the layout's topbar.
 * On mobile the topbar also houses the burger menu button.
 */
let content: Snippet<[]> | null = $state(null);

export function getTopbarContent(): Snippet<[]> | null {
    return content;
}

export function setTopbarContent(c: Snippet<[]> | null) {
    content = c;
}
