/**
 * Composable abort context for Svelte components.
 *
 * Each component can create its own abort scope that is automatically linked
 * to its parent's scope. If the parent aborts, all children abort too — but a
 * child can abort independently without affecting siblings.
 *
 * Typical usage in a component:
 *
 * ```svelte
 * <script lang="ts">
 *   import { createAbortContext, getAbortContext } from '$lib/abort';
 *
 *   const { signal, abort } = createAbortContext();
 *
 *   // `signal` aborts when `abort()` is called OR when the parent aborts.
 * </script>
 * ```
 */

import { setContext, getContext } from 'svelte';

const KEY = Symbol('abort-signal');

/**
 * Create a new AbortController whose signal is composed with the parent
 * context's signal (if any).
 *
 * Returns the controller so the caller can both read the merged signal and
 * trigger an abort. The merged signal is also stored in Svelte's context
 * for nested components to pick up.
 */
export function createAbortContext(): AbortController {
    const controller = new AbortController();
    const parent = getParentSignal();

    if (parent) {
        if (parent.aborted) {
            // Parent has already aborted; the child should start aborted too.
            controller.abort();
        } else {
            // If the parent aborts, abort this controller too.
            parent.addEventListener('abort', () => controller.abort(), { once: true });
        }
    }

    // Store the controller's signal for children. If the parent was already
    // stored, this overwrites it — which is correct because children should
    // compose with *this* component's signal, not the grandparent's.
    setContext(KEY, controller.signal);

    return controller;
}

/**
 * Set an externally-created signal as the abort context, composing it with
 * any parent context signal.
 *
 * Useful when you already have an AbortSignal (e.g. from a prop) and want
 * to make it available to children, merged with the parent scope.
 *
 * Returns the composed signal.
 */
export function setAbortContext(signal: AbortSignal): AbortSignal {
    const parent = getParentSignal();

    if (parent) {
        signal = AbortSignal.any([signal, parent]);
    }

    return setContext(KEY, signal);
}

/**
 * Retrieve the current abort signal from the nearest ancestor component
 * that called `createAbortContext()` or `setAbortContext()`.
 *
 * Returns `undefined` if no ancestor has set an abort context.
 */
export function getAbortContext(): AbortSignal | undefined {
    return getParentSignal();
}

/**
 * Create a new AbortController that automatically aborts when the parent
 * signal fires. If `parent` is undefined or already aborted, the controller
 * is aborted immediately.
 *
 * Useful inside `$effect` blocks where you need an effect-scoped controller
 * that also responds to a component-level (or ancestor) abort context.
 */
export function linkedController(parent?: AbortSignal): AbortController {
    const controller = new AbortController();
    if (parent) {
        if (parent.aborted) {
            controller.abort();
        } else {
            parent.addEventListener('abort', () => controller.abort(), { once: true });
        }
    }
    return controller;
}

/** Internal helper — reads the parent signal without throwing if context is missing. */
function getParentSignal(): AbortSignal | undefined {
    return getContext<AbortSignal | undefined>(KEY);
}
