import type { Action } from 'svelte/action';

export interface AutoscrollOptions {
    /** Scroll to bottom on initial mount, regardless of position.
     *  Default true. Set false for "stay where you are on mount"
     *  (e.g. log viewers where the user should see the start). */
    scrollOnMount?: boolean;
}

/** Svelte action that follows streaming content.
 *
 * Attach to the element whose *content* streams (the body that gains
 * text). The action scrolls the element that actually scrolls that
 * content — ``node`` itself if it's a scroll container, otherwise the
 * nearest scrollable ancestor, otherwise the page. This lets one
 * action drive an inner scroll container (e.g. ``ReasoningCard``'s
 * ``<pre>``) AND an area-level scroll where the streaming element has
 * no own overflow (e.g. the prompt preview on mobile, where the whole
 * content area scrolls together rather than an inner "box").
 *
 * Behaviour:
 * - On mount (when ``scrollOnMount`` is true), jumps to the bottom.
 * - On every content change (``MutationObserver`` on ``node``), scrolls
 *   to the bottom if auto-scroll is enabled.
 * - Any user-initiated scroll-up (mouse wheel up, touch drag up,
 *   keyboard scroll up — anything that decreases ``scrollTop``)
 *   pauses auto-scroll. Auto-scroll resumes only when the user scrolls
 *   back to (or within a few pixels of) the bottom.
 *
 * The scroll-up detection compares ``scrollTop`` across scroll events
 * on the resolved target: our own programmatic scroll only ever sets
 * ``scrollTop = scrollHeight`` (max), which can only stay the same or
 * increase, so a decrease is unambiguously user-initiated.
 *
 * The resolved target is re-evaluated on viewport resize, because a
 * responsive layout can move which element scrolls (inner container on
 * desktop vs. the content area on mobile) and the listener must follow.
 */
const RESUME_THRESHOLD = 5; // px from bottom at which auto-scroll resumes

function overflowsVertically(el: HTMLElement): boolean {
    return /auto|scroll|overlay/.test(getComputedStyle(el).overflowY);
}

/** The element whose ``scrollTop`` reflects scrolling ``node``'s
 *  content: ``node`` if it scrolls, else the nearest scrollable
 *  ancestor, else the page (``document.scrollingElement``). */
function resolveScrollTarget(node: HTMLElement): HTMLElement {
    if (overflowsVertically(node)) {
        return node;
    }
    let el: HTMLElement | null = node.parentElement;
    while (el) {
        if (overflowsVertically(el)) {
            return el;
        }
        el = el.parentElement;
    }
    return (document.scrollingElement as HTMLElement | null) ?? document.documentElement;
}

export const autoscroll: Action<HTMLElement, AutoscrollOptions | undefined> = (node, options) => {
    let scrollOnMount = options?.scrollOnMount ?? true;
    let userScrolledUp = false;
    let target = resolveScrollTarget(node);
    let prevScrollTop = target.scrollTop;

    // ``scroll`` doesn't bubble, so listen on the target itself. When the
    // target is the page (the scrollingElement), the scroll event fires
    // on ``window`` instead.
    function scrollEventTarget(): HTMLElement | Window {
        return target === document.scrollingElement ? window : target;
    }

    function onScroll() {
        const currentScrollTop = target.scrollTop;
        // Any decrease in scrollTop between events = user scrolled up.
        if (currentScrollTop < prevScrollTop) {
            userScrolledUp = true;
        }
        prevScrollTop = currentScrollTop;

        // Resume only when the user has come back to the bottom.
        if (userScrolledUp) {
            const distFromBottom = target.scrollHeight - currentScrollTop - target.clientHeight;
            if (distFromBottom <= RESUME_THRESHOLD) {
                userScrolledUp = false;
            }
        }
    }

    function maybeScroll() {
        if (userScrolledUp) {
            return;
        }
        target.scrollTop = target.scrollHeight;
    }

    const observer = new MutationObserver(maybeScroll);
    observer.observe(node, { childList: true, characterData: true, subtree: true });

    let eventTarget: HTMLElement | Window = scrollEventTarget();
    eventTarget.addEventListener('scroll', onScroll, { passive: true });

    // Re-resolve on resize — a responsive layout can switch which
    // element actually scrolls, so the listener may need to move.
    function onResize() {
        const next = resolveScrollTarget(node);
        if (next === target) {
            return;
        }
        target = next;
        prevScrollTop = target.scrollTop;
        const nextEventTarget = scrollEventTarget();
        if (nextEventTarget !== eventTarget) {
            eventTarget.removeEventListener('scroll', onScroll);
            eventTarget = nextEventTarget;
            eventTarget.addEventListener('scroll', onScroll, { passive: true });
        }
    }
    window.addEventListener('resize', onResize);

    if (scrollOnMount) {
        // Defer to next frame so the freshly-rendered DOM has its
        // layout computed (scrollHeight etc. need a layout pass).
        requestAnimationFrame(() => {
            target.scrollTop = target.scrollHeight;
        });
    }

    return {
        update(newOptions) {
            scrollOnMount = newOptions?.scrollOnMount ?? true;
        },
        destroy() {
            observer.disconnect();
            eventTarget.removeEventListener('scroll', onScroll);
            window.removeEventListener('resize', onResize);
        }
    };
};
