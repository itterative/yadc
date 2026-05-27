import { writable } from 'svelte/store';
import { PasswordRequiredError } from '$lib/api';
import { sessionPassword } from './sessionPassword';

interface PasswordPromptState {
    open: boolean;
    resolve: ((password: string) => void) | null;
    reject: ((reason: PasswordPromptCancelled) => void) | null;
}

let state: PasswordPromptState = { open: false, resolve: null, reject: null };

/** Shared promise when a password prompt is already open.  All concurrent
 *  callers await the same promise so the user is only prompted once. */
let pendingPromise: Promise<string> | null = null;

const _store = writable<boolean>(false);

/** Svelte store that tracks whether the password prompt dialog is open. */
export const passwordPromptOpen = { subscribe: _store.subscribe };

/** Show the password prompt and return a Promise that resolves with the
 *  entered password or rejects if the user cancels.
 *
 *  If a prompt is already open the existing promise is returned so every
 *  concurrent caller waits for the same user input. */
export function requestPassword(): Promise<string> {
    if (pendingPromise) {
        return pendingPromise;
    }

    pendingPromise = new Promise((resolve, reject) => {
        state = { open: true, resolve, reject };
        _store.set(true);
    });

    return pendingPromise;
}

/** Resolve the pending password request and close the dialog. */
export function submitPassword(password: string) {
    const resolver = state.resolve;
    state.resolve = null;
    state.reject = null;
    state.open = false;
    _store.set(false);
    pendingPromise = null;
    resolver?.(password);
}

/** Reject the pending password request and close the dialog. */
export function cancelPassword() {
    const rejecter = state.reject;
    state.resolve = null;
    state.reject = null;
    state.open = false;
    _store.set(false);
    pendingPromise = null;
    rejecter?.(new PasswordPromptCancelled());
}

/** Thrown when the user cancels the password prompt. */
export class PasswordPromptCancelled extends Error {
    constructor() {
        super('Password prompt cancelled');
        this.name = 'PasswordPromptCancelled';
    }
}

/** Try *action* once.  If it fails with `PasswordRequiredError`, clear any
 *  cached password, prompt the user, save the entered password to
 *  `sessionPassword`, and retry exactly once.
 *
 *  If another prompt is already open when the password is needed,
 *  `requestPassword()` returns the existing promise automatically so the
 *  user is only prompted once. */
export async function withPasswordRetry<T>(action: () => Promise<T>): Promise<T> {
    try {
        return await action();
    } catch (e) {
        if (!(e instanceof PasswordRequiredError)) {
            throw e;
        }
        sessionPassword.clear();
    }

    const password = await requestPassword();
    if (!password) {
        throw new PasswordPromptCancelled();
    }
    sessionPassword.set(password);

    try {
        return await action();
    } catch (e) {
        if (e instanceof PasswordRequiredError) {
            sessionPassword.clear();
        }
        throw e;
    }
}
