import { writable } from 'svelte/store';
import { API_BASE, apiErrorMessage, PasswordRequiredError } from '$lib/api';

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

/** Set the ``yadc_password`` session cookie by POSTing to ``/api/auth/password``.
 *
 *  The backend validates the password by attempting to decrypt the
 *  password-mode private key. A 403 means the password is wrong —
 *  callers should surface this as a "password is incorrect" error
 *  (typically by throwing a ``PasswordRequiredError``). 204 means the
 *  cookie was set and subsequent API calls will carry it automatically.
 *
 *  Exported so other modules (e.g. ``SecuritySettings``) can refresh
 *  the cookie after the user changes their key storage password,
 *  without going through the prompt flow. */
export async function setAuthCookie(password: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/auth/password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password })
    });
    if (res.status === 403) {
        throw new PasswordRequiredError('Password is incorrect.');
    }
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

/** Clear the ``yadc_password`` session cookie via ``DELETE /api/auth/password``.
 *
 *  Idempotent — 204 whether or not the cookie was set. Use this when
 *  the user switches to keyring mode (the password is no longer
 *  needed) or signs out of password mode entirely. */
export async function clearAuthCookie(): Promise<void> {
    await fetch(`${API_BASE}/api/auth/password`, { method: 'DELETE' });
}

/** Try *action* once.  If it fails with `PasswordRequiredError`, clear
 *  any existing ``yadc_password`` cookie, prompt the user, set the
 *  cookie from their input, and retry exactly once.
 *
 *  If the cookie set itself fails with 403 (wrong password), the
 *  ``PasswordRequiredError`` propagates so the caller can show a
 *  "password is incorrect" message rather than retrying indefinitely.
 *
 *  If another prompt is already open when the password is needed,
 *  `requestPassword()` returns the existing promise automatically so the
 *  user is only prompted once.
 *
 *  Abort handling: when *signal* aborts, the wrapper throws
 *  ``DOMException('aborted', 'AbortError')`` at the next check point
 *  (before each ``await``). If the signal aborts while the prompt is
 *  open, the prompt is dismissed (``cancelPassword()``) so the user
 *  isn't left staring at a dialog for a request that no longer matters.
 *  The signal is **not** forwarded to *action* — *action* is expected
 *  to close over its own signal and pass it to ``fetch`` (or the
 *  equivalent). The inner action's own abort error propagates as-is. */
export async function withPasswordRetry<T>(
    action: () => Promise<T>,
    signal?: AbortSignal
): Promise<T> {
    const throwIfAborted = () => {
        if (signal?.aborted) {
            throw new DOMException('The operation was aborted.', 'AbortError');
        }
    };

    // While the prompt is open, aborts from the caller dismiss it so the
    // user isn't left waiting for a request that no longer matters. The
    // listener is removed in the ``finally`` block below.
    const onAbort = () => cancelPassword();
    signal?.addEventListener('abort', onAbort, { once: true });

    try {
        try {
            throwIfAborted();
            return await action();
        } catch (e) {
            if (!(e instanceof PasswordRequiredError)) {
                throw e;
            }
            throwIfAborted();
            // Clear any stale cookie so the prompt's setAuthCookie
            // isn't shadowed by an old bad value.
            await clearAuthCookie();
        }

        let password: string | null;
        try {
            password = await requestPassword();
        } catch (e) {
            // ``requestPassword()`` rejects with ``PasswordPromptCancelled``
            // when the user clicks cancel *or* when our onAbort listener
            // calls ``cancelPassword()`` after the signal aborts. Translate
            // the abort case to ``AbortError`` so the caller's signal is
            // honored consistently.
            if (e instanceof PasswordPromptCancelled && signal?.aborted) {
                throw new DOMException('The operation was aborted.', 'AbortError');
            }
            throw e;
        }
        if (!password) {
            throw new PasswordPromptCancelled();
        }
        throwIfAborted();

        // Throws PasswordRequiredError on 403 — the action below never
        // runs, and the caller shows the existing "incorrect password"
        // toast / error message.
        await setAuthCookie(password);

        try {
            throwIfAborted();
            return await action();
        } catch (e) {
            if (e instanceof PasswordRequiredError) {
                await clearAuthCookie();
            }
            throw e;
        }
    } finally {
        signal?.removeEventListener('abort', onAbort);
    }
}
