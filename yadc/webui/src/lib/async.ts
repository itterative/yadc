import { browser } from '$app/environment';

const DEFAULT_DEBOUNCE_MS = Number(import.meta.env.PUBLIC_API_DEFAULT_DEBOUNCE_MS) || 25;

export interface DebounceOptions {
    /** Debounce delay in milliseconds. Defaults to `PUBLIC_API_DEFAULT_DEBOUNCE_MS` or 25. */
    delay?: number;
    /** If true, emit console.log diagnostics for debugging. */
    log?: boolean;
}

type NoopLogger = (...args: unknown[]) => void;

/**
 * Debounce an async function keyed by its arguments.
 *
 * Each unique set of arguments (serialized via `JSON.stringify`) gets its own
 * timer.  Subsequent calls with the same arguments **reset** the timer (true
 * debounce).  While a call is pending or in-flight, duplicate calls return a
 * new promise backed by the same underlying fetch (dedupe).  When the callback
 * completes the entry is removed so the next call starts fresh.
 *
 * **AbortSignal handling** (opt-in by convention): if the **last** argument
 * is an `AbortSignal`, the debouncer treats it as the caller's signal.  The
 * debouncer owns a per-entry `AbortController` and passes `controller.signal`
 * to the underlying `cb(...)` instead of the caller's signal.  Each caller's
 * signal is tracked independently: if a caller aborts, only that caller's
 * promise is rejected and their interest in the fetch is dropped.  The fetch
 * continues for any remaining interested callers and is only cancelled when
 * the last interested caller aborts.  Callers that pass an already-aborted
 * signal are rejected immediately without scheduling a fetch.
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
    options?: DebounceOptions
): (...args: Parameters<T>) => Promise<Awaited<ReturnType<T>>> {
    const delay = options?.delay ?? DEFAULT_DEBOUNCE_MS;
    const log: NoopLogger = options?.log ? console.log.bind(console) : () => {};

    if (!browser) {
        // SSR: no timers available — pass through directly.
        return cb as (...args: Parameters<T>) => Promise<Awaited<ReturnType<T>>>;
    }

    type V = Awaited<ReturnType<T>>;

    interface Caller {
        /** The promise given to this specific caller. */
        promise: Promise<V>;
        resolve: (v: V) => void;
        reject: (e: unknown) => void;
        /** The caller's signal, if any. Used to drop interest on abort. */
        signal?: AbortSignal;
        /** Cleanup for the abort listener registered on `signal`. */
        cleanupSignal?: () => void;
    }

    interface Entry {
        /** The dedup key — stored so `finalize` can remove the entry from the
         *  map without scanning it. */
        key: string;
        /** Eagerly-created AbortController for the in-flight call.  Its signal
         *  is passed to `cb(...)` when the timer fires.  Aborting this
         *  controller aborts the underlying fetch. */
        controller: AbortController;
        /** Callers interested in the result of this debounced call. */
        callers: Caller[];
        /** Whether any interested caller passed an `AbortSignal`.  If true, the
         *  callback is invoked with the entry's controller signal so that
         *  aborts are forwarded to the underlying fetch. */
        hadSignal: boolean;
        /** True once `fire()` has been invoked for this entry.  Used so later
         *  deduped callers during the in-flight phase don't start another
         *  timer / another fetch. */
        fired: boolean;
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

        // Dedupe: reuse the existing entry if one is already pending or
        // in-flight. Each caller still gets their own promise so that one
        // caller's abort does not reject unrelated callers.
        let entry = entries.get(key);
        if (entry === undefined) {
            entry = {
                key,
                controller: new AbortController(),
                callers: [],
                hadSignal: false,
                fired: false
            };
            entries.set(key, entry);
            log('[debounce]', key, 'created entry');
        } else {
            log(
                '[debounce]',
                key,
                'reused entry, callers=',
                entry.callers.length,
                'fired=',
                entry.fired,
                'ctrlAborted=',
                entry.controller.signal.aborted
            );
        }

        let resolve!: (value: V) => void;
        let reject!: (reason: unknown) => void;
        const promise = new Promise<V>((res, rej) => {
            resolve = res;
            reject = rej;
        });
        const caller: Caller = { promise, resolve, reject, signal: callerSignal };
        entry.callers.push(caller);

        // Link the caller's signal to the entry: if this caller aborts, drop
        // their interest and reject their promise. Only abort the underlying
        // controller when no interested callers remain.
        if (callerSignal !== undefined) {
            entry.hadSignal = true;
            const onAbort = () => {
                log('[debounce]', key, 'caller abort, callersBefore=', entry!.callers.length);
                removeCaller(entry!, caller);
                caller.reject(caller.signal?.reason ?? new DOMException('Aborted', 'AbortError'));
                if (entry!.callers.length === 0) {
                    log('[debounce]', key, 'last caller aborted, finalizing');
                    entry!.controller.abort();
                    finalize(entry!);
                }
            };
            callerSignal.addEventListener('abort', onAbort, { once: true });
            caller.cleanupSignal = () => callerSignal!.removeEventListener('abort', onAbort);
        }

        // Start / restart the debounce timer, unless the entry is already
        // in-flight (timer already fired).  cbArgs comes from the latest
        // caller; hadSignal is OR-ed across every deduped caller for this key.
        if (!entry.fired) {
            const existingTimer = timers.get(key);
            if (existingTimer !== undefined) {
                log('[debounce]', key, 'reset timer');
                window.clearTimeout(existingTimer);
            }
            const timer = window.setTimeout(() => {
                timers.delete(key);
                const current = entries.get(key);
                if (current === undefined || current.callers.length === 0) {
                    // All callers aborted before the timer fired — clean up the
                    // stale entry without invoking the callback.
                    log('[debounce]', key, 'timer fired, no callers');
                    if (current !== undefined) {
                        finalize(current);
                    }
                    return;
                }
                log('[debounce]', key, 'timer fired, calling cb');
                fire(current, cbArgs);
            }, delay);
            timers.set(key, timer);
        } else {
            log('[debounce]', key, 'already in-flight, not scheduling new timer');
        }

        return promise;
    };

    function fire(entry: Entry, cbArgs: unknown[]) {
        entry.fired = true;
        log(
            '[debounce]',
            entry.key,
            'fire, hadSignal=',
            entry.hadSignal,
            'ctrlAborted=',
            entry.controller.signal.aborted
        );
        try {
            // Append the controller's signal as the last positional arg of
            // `cb` per the AbortSignal convention.  TypeScript can't verify
            // this statically, so we cast through `unknown[]` to satisfy the
            // generic constraint.
            const callArgs: unknown[] = entry.hadSignal
                ? [...cbArgs, entry.controller.signal]
                : cbArgs;
            (cb as (...a: unknown[]) => Promise<V>)(...callArgs)
                .then((v) => {
                    log('[debounce]', entry.key, 'cb resolved');
                    const interested = entry.callers;
                    finalize(entry);
                    for (const c of interested) {
                        c.resolve(v);
                    }
                })
                .catch((e) => {
                    const interested = entry.callers;
                    finalize(entry);
                    if (interested.length === 0) {
                        // No callers are waiting anymore (they all aborted).
                        // Swallow the rejection rather than logging a spurious
                        // 'cb rejected' for a fetch nobody cares about.
                        log('[debounce]', entry.key, 'cb rejected, no interested callers');
                        return;
                    }
                    log('[debounce]', entry.key, 'cb rejected', e);
                    for (const c of interested) {
                        c.reject(e);
                    }
                });
        } catch (e) {
            const interested = entry.callers;
            finalize(entry);
            for (const c of interested) {
                c.reject(e);
            }
        }
    }

    function removeCaller(entry: Entry, caller: Caller) {
        const index = entry.callers.indexOf(caller);
        if (index >= 0) {
            entry.callers.splice(index, 1);
        }
        caller.cleanupSignal?.();
        caller.cleanupSignal = undefined;
    }

    function finalize(entry: Entry) {
        log('[debounce]', entry.key, 'finalize');
        // Only delete from the map if this exact entry is still the one stored
        // under the key. A newer entry may have been created for the same key
        // while this one was still in-flight; we must not clobber it.
        if (entries.get(entry.key) === entry) {
            entries.delete(entry.key);
        }
        for (const c of entry.callers) {
            c.cleanupSignal?.();
            c.cleanupSignal = undefined;
        }
        entry.callers = [];
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
