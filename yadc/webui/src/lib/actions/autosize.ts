import type { Action } from 'svelte/action';

export interface AutosizeOptions {
    /** Max height in pixels. Past this, the textarea scrolls. Default 360. */
    maxHeight?: number;
}

/** Svelte action that resizes a textarea to fit its content.
 *
 * Sets the height to ``scrollHeight`` (capped at ``maxHeight``). Re-runs
 * on input events so typing, pasting, and backspace keep the textarea
 * in sync. Initial mount is also handled, so no companion ``oninput``
 * handler is needed in the parent.
 */
export const autosize: Action<HTMLTextAreaElement, AutosizeOptions | undefined> = (
    node,
    options
) => {
    let maxHeight = options?.maxHeight ?? 360;

    function resize() {
        node.style.height = 'auto';
        node.style.height = Math.min(node.scrollHeight, maxHeight) + 'px';
    }

    resize();
    const handler = () => resize();
    node.addEventListener('input', handler);

    return {
        update(newOptions) {
            maxHeight = newOptions?.maxHeight ?? 360;
            resize();
        },
        destroy() {
            node.removeEventListener('input', handler);
        }
    };
};
