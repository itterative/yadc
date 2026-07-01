/** Shared positioning for native HTML ``popover="auto"`` menus anchored to
 *  a trigger element. Used by :comp:`TagChip` (category-reassignment menu)
 *  and :comp:`TagInput` (suggestion dropdown) so the fixed-position clamping
 *  logic lives in one place rather than being copy-pasted. */

/** Clamp a fixed-position popover below (or flipped above) its anchor and
 *  shift it left if it would overflow the viewport's right edge. Writes the
 *  computed ``top`` / ``left`` / ``min-width`` styles directly onto ``menu``
 *  and ensures it's shown. Returns ``false`` when the anchor has scrolled
 *  entirely out of the viewport so the caller can hide the menu. */
export function positionPopover(
    menu: HTMLElement,
    anchor: HTMLElement,
    options: { padding?: number; minWidthFloor?: number } = {}
): boolean {
    const { padding = 4, minWidthFloor = 120 } = options;
    const rect = anchor.getBoundingClientRect();
    if (rect.bottom < 0 || rect.top > window.innerHeight) {
        return false;
    }

    menu.style.minWidth = `${Math.max(Math.round(rect.width), minWidthFloor)}px`;
    if (!menu.matches(':popover-open')) {
        menu.showPopover();
    }

    let top = rect.bottom + padding;
    let left = rect.left;

    // Flip above the anchor when there's no room below.
    const ddRect = menu.getBoundingClientRect();
    if (top + ddRect.height > window.innerHeight - padding) {
        top = Math.max(rect.top - ddRect.height - padding, padding);
    }
    // Shift left when it would overflow the right edge.
    if (left + ddRect.width > window.innerWidth - padding) {
        left = Math.max(window.innerWidth - ddRect.width - padding, padding);
    }

    menu.style.top = `${Math.round(top)}px`;
    menu.style.left = `${Math.round(left)}px`;
    return true;
}

/** Whether the anchor is still on-screen enough to keep the menu open
 *  (used by scroll handlers that close menus whose anchor scrolled away). */
export function anchorVisible(anchor: HTMLElement): boolean {
    return anchor.getBoundingClientRect().width !== 0;
}
