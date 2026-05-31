/**
 * Promise-based confirmation dialog store.
 *
 * Provides a reactive dialog state and a `confirm()` function that returns
 * a Promise<boolean>. Resolves `true` when the user confirms, `false` when
 * they cancel (or press Escape). Mounted as a single global instance in
 * +layout.svelte via ConfirmDialog.svelte.
 *
 * Usage (simple string):
 *   import { confirmDialog } from '$lib/stores/confirm';
 *   const ok = await confirmDialog.danger('Delete dataset "foo"?');
 *
 * Usage (with snippet body):
 *   {#snippet deleteBody()}
 *     <p>Delete <strong>{name}</strong>?</p>
 *   {/snippet}
 *   <button onclick={() => confirmDialog.danger({ body: deleteBody })}>Delete</button>
 *
 * Usage (general):
 *   const ok = await confirmDialog.confirm({
 *     title: 'Overwrite',
 *     message: 'Overwrite existing captions?',
 *     variant: 'warning',
 *     confirmLabel: 'Overwrite',
 *   });
 */

import { writable, readonly, type Readable } from 'svelte/store';
import type { Snippet } from 'svelte';

// --- Types ---

export type ConfirmVariant = 'danger' | 'warning' | 'info';

export interface ConfirmOptions {
    /** Dialog title. Defaults vary by variant. */
    title?: string;
    /** Simple text message. Mutually exclusive with `body`. */
    message?: string;
    /** Rich snippet body. Takes precedence over `message`. */
    body?: Snippet;
    /** Visual variant. Default: `'warning'`. */
    variant?: ConfirmVariant;
    /** Label for the confirm button. Defaults vary by variant. */
    confirmLabel?: string;
    /** Label for the cancel button. Default: `'Cancel'`. */
    cancelLabel?: string;
}

export interface ConfirmState {
    open: boolean;
    options: ConfirmOptions | null;
}

// --- Internal state ---

interface InternalState extends ConfirmState {
    resolve: ((value: boolean) => void) | null;
}

let state: InternalState = { open: false, options: null, resolve: null };

const _store = writable<ConfirmState>({ open: false, options: null });

/** Reactive state for the global ConfirmDialog component. */
export const confirmState: Readable<ConfirmState> = readonly(_store);

// --- Actions ---

/** Show a confirmation dialog and return a Promise that resolves to
 *  `true` (confirm) or `false` (cancel/escape). If a dialog is already
 *  open, it is dismissed with `false` first. */
export function confirm(options: ConfirmOptions): Promise<boolean> {
    // Dismiss any existing dialog
    if (state.open && state.resolve) {
        state.resolve(false);
    }

    return new Promise<boolean>((resolve) => {
        state = { open: true, options, resolve };
        _store.set({ open: true, options });
    });
}

/** Resolve the current dialog. Called by ConfirmDialog.svelte. */
export function resolveConfirm(value: boolean): void {
    const resolver = state.resolve;
    state = { open: false, options: null, resolve: null };
    _store.set({ open: false, options: null });
    resolver?.(value);
}

// --- Convenience helpers ---

const DEFAULT_TITLES: Record<ConfirmVariant, string> = {
    danger: 'Confirm Delete',
    warning: 'Warning',
    info: 'Confirm'
};

const DEFAULT_CONFIRM_LABELS: Record<ConfirmVariant, string> = {
    danger: 'Delete',
    warning: 'Confirm',
    info: 'OK'
};

type StringOrOptions = string | ConfirmOptions;

function withDefaults(variant: ConfirmVariant, input: StringOrOptions): ConfirmOptions {
    if (typeof input === 'string') {
        return {
            message: input,
            variant,
            title: DEFAULT_TITLES[variant],
            confirmLabel: DEFAULT_CONFIRM_LABELS[variant]
        };
    }
    return {
        ...input,
        variant,
        title: input.title ?? DEFAULT_TITLES[variant],
        confirmLabel: input.confirmLabel ?? DEFAULT_CONFIRM_LABELS[variant]
    };
}

export const confirmDialog = {
    /** General confirm with full options. */
    confirm(options: ConfirmOptions): Promise<boolean> {
        return confirm(options);
    },

    /** Destructive action confirm (red). */
    danger(input: StringOrOptions): Promise<boolean> {
        return confirm(withDefaults('danger', input));
    },

    /** Warning confirm (yellow). */
    warning(input: StringOrOptions): Promise<boolean> {
        return confirm(withDefaults('warning', input));
    },

    /** Informational confirm (blue/accent). */
    info(input: StringOrOptions): Promise<boolean> {
        return confirm(withDefaults('info', input));
    }
};
