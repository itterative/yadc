import { browser } from '$app/environment';

const DEFAULT_DEBOUNCE_MS = Number(import.meta.env.PUBLIC_API_DEFAULT_DEBOUNCE_MS) || 25;

/**
 * Debounce an async function keyed by its arguments.
 *
 * Each unique set of arguments (serialized via `JSON.stringify`) gets its own
 * timer.  Subsequent calls with the same arguments **reset** the timer (true
 * debounce).  While a call is pending or in-flight, duplicate calls return the
 * existing promise (dedupe).  When the callback completes the entry is removed
 * so the next call starts fresh.
 *
 * **AbortSignal handling** (opt-in by convention): if the **last** argument
 * is an `AbortSignal`, the debouncer treats it as the caller's signal.  The
 * debouncer owns a per-entry `AbortController` (created eagerly on the first
 * call) and passes `controller.signal` to the underlying `cb(...)` instead of
 * the caller's signal.  Every caller's signal is linked to the controller via
 * an abort event listener — if **any** linked signal aborts, the in-flight
 * call is aborted.  All listeners are removed when the entry resolves or
 * rejects.  Callers that pass an already-aborted signal are rejected
 * immediately without scheduling a fetch.
 *
 * If multiple callers dedupe for the same key and *any* of them passed a
 * signal, the callback is invoked with the debouncer's internal controller
 * signal so that aborts from any linked caller can reach the underlying
 * fetch.
 *
 * The convention requires the wrapped callback's last parameter to be an
 * `AbortSignal` (typically optional) whenever the caller passes a signal.
 * Callbacks that don't accept a signal must be called without one.
 *
 * On the server (SSR) the wrapper is a no-op pass-through since `window` is
 * not available.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function debounce<T extends (...args: any[]) => Promise<any>>(
    cb: T,
    delay: number = DEFAULT_DEBOUNCE_MS
): (...args: Parameters<T>) => Promise<Awaited<ReturnType<T>>> {
    if (!browser) {
        // SSR: no timers available — pass through directly.
        return cb as (...args: Parameters<T>) => Promise<Awaited<ReturnType<T>>>;
    }

    type V = Awaited<ReturnType<T>>;

    interface Entry {
        /** The dedup key — stored so `finalize` can remove the entry from the
         *  map without scanning it. */
        key: string;
        resolve: (v: V) => void;
        reject: (e: unknown) => void;
        promise: Promise<V>;
        /** Eagerly-created AbortController for the in-flight call.  Its signal
         *  is passed to `cb(...)` when the timer fires.  Aborting this
         *  controller aborts the underlying fetch. */
        controller: AbortController;
        /** Abort event listener cleanups, called when the entry resolves or
         *  rejects. */
        cleanups: Array<() => void>;
        /** Whether any deduped caller passed an `AbortSignal`.  If true, the
         *  callback is invoked with the entry's controller signal so that
         *  aborts are forwarded to the underlying fetch. */
        hadSignal: boolean;
    }

    const timers = new Map<string, number>();
    const entries = new Map<string, Entry>();

    return (...args: Parameters<T>) => {
        // Convention: the last arg, if it is an AbortSignal, is the caller's
        // signal.  It is used for abort coordination but is NOT passed to
        // `cb` directly — the debouncer passes `entry.controller.signal`
        // instead.
        let callerSignal: AbortSignal | undefined;
        let cbArgs: unknown[];
        if (args.length > 0 && args[args.length - 1] instanceof AbortSignal) {
            callerSignal = args[args.length - 1] as AbortSignal;
            cbArgs = args.slice(0, -1);
        } else {
            cbArgs = args.slice();
        }

        // If the caller's signal is already aborted, reject immediately.
        if (callerSignal?.aborted) {
            return Promise.reject(callerSignal.reason ?? new DOMException('Aborted', 'AbortError'));
        }

        // The key is computed from the args without the signal so that calls
        // with the same non-signal arguments dedupe regardless of whether a
        // signal was passed.
        const key = JSON.stringify(cbArgs);

        // Cancel and restart the timer (true debounce — reset on each call).
        const existingTimer = timers.get(key);
        if (existingTimer !== undefined) {
            window.clearTimeout(existingTimer);
        }

        // Dedupe: reuse the existing entry if one is already pending or
        // in-flight.
        let entry = entries.get(key);
        if (entry === undefined) {
            let resolve!: (value: V) => void;
            let reject!: (reason: unknown) => void;
            // The Promise executor runs synchronously (JS spec), so resolve/reject
            // are guaranteed to be assigned before we use them below.
            const promise = new Promise<V>((res, rej) => {
                resolve = res;
                reject = rej;
            });
            entry = {
                key,
                resolve,
                reject,
                promise,
                controller: new AbortController(),
                cleanups: [],
                hadSignal: false
            };
            entries.set(key, entry);
        }

        // Link the caller's signal to the entry's controller: if the caller's
        // signal aborts, abort the in-flight call (if/when it starts).
        if (callerSignal !== undefined) {
            entry.hadSignal = true;
            const onAbort = () => entry!.controller.abort();
            callerSignal.addEventListener('abort', onAbort, { once: true });
            entry.cleanups.push(() => callerSignal!.removeEventListener('abort', onAbort));
        }

        // Start / restart the debounce timer.  cbArgs comes from the latest
        // caller; hadSignal is OR-ed across every deduped caller for this key.
        const timer = window.setTimeout(() => {
            timers.delete(key);
            const current = entries.get(key);
            if (current === undefined) {
                return;
            }
            fire(current, cbArgs, current.hadSignal);
        }, delay);
        timers.set(key, timer);

        return entry.promise;
    };

    function fire(entry: Entry, cbArgs: unknown[], hadSignal: boolean) {
        try {
            // Append the controller's signal as the last positional arg of
            // `cb` per the AbortSignal convention.  TypeScript can't verify
            // this statically, so we cast through `unknown[]` to satisfy the
            // generic constraint.
            const callArgs: unknown[] = hadSignal ? [...cbArgs, entry.controller.signal] : cbArgs;
            (cb as (...a: unknown[]) => Promise<V>)(...callArgs)
                .then((v) => {
                    finalize(entry);
                    entry.resolve(v);
                })
                .catch((e) => {
                    finalize(entry);
                    entry.reject(e);
                });
        } catch (e) {
            finalize(entry);
            entry.reject(e);
        }
    }

    function finalize(entry: Entry) {
        entries.delete(entry.key);
        for (const cleanup of entry.cleanups) {
            cleanup();
        }
        entry.cleanups.length = 0;
    }
}

/**
 * Wraps an async function so calls are serialized — each invocation
 * waits for the previous one to complete before starting.
 */
export function synchronized<R, T extends (...args: unknown[]) => Promise<R>>(cb: T) {
    const promises: Promise<R>[] = [];

    return async (...args: Parameters<T>) => {
        if (promises.length) {
            await Promise.all(promises);
        }

        const promise = cb(...args);
        promises.push(promise);

        try {
            return await promise;
        } finally {
            const promiseIndex = promises.indexOf(promise);
            if (promiseIndex >= 0) {
                promises.splice(promiseIndex, 1);
            }
        }
    };
}

/** Sleep for the given number of seconds. */
export function sleep(delay: number): Promise<void> {
    if (!browser) {
        return Promise.resolve();
    }

    return new Promise((resolve) => window.setTimeout(resolve, delay * 1000));
}

/**
 * Delays execution of an async callback by `delay` seconds.
 */
export function delayed<R, T extends (...args: unknown[]) => Promise<R>>(cb: T, delay: number) {
    return async (...args: Parameters<T>) => {
        await sleep(delay);
        await cb(...args);
    };
}

/**
 * Returns a debounced wrapper around a synchronous callback.
 *
 * Each call resets the timer.  The callback only executes after `delay`
 * milliseconds have elapsed without any new calls.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function deferred<T extends (...args: any[]) => void>(
    cb: T,
    delay: number = 10
): (...args: Parameters<T>) => void {
    let cbTimeout: number | null = null;

    return (...args: Parameters<T>) => {
        if (cbTimeout) {
            window.clearTimeout(cbTimeout);
        }

        cbTimeout = window.setTimeout(() => cb(...args), delay);
    };
}
