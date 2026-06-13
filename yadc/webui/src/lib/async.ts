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
 * Each unique set of arguments (serialized via `JSON.stringify`) gets its
 * own entry. Subsequent calls with the same arguments **reset** the
 * debounce timer (trailing debounce). Once the timer fires, the callback is
 * invoked once; later calls during the in-flight phase dedupe onto the
 * in-flight entry.
 *
 * **AbortSignal handling** (opt-in by convention): if the last argument is
 * an `AbortSignal`, it is treated as the caller's signal. The debouncer
 * passes the entry's controller signal to `cb` (not the caller's signal),
 * so aborts are forwarded to the underlying fetch. Each caller's signal is
 * tracked independently: if a caller aborts, only that caller's promise is
 * rejected; the fetch continues for remaining callers and is cancelled
 * only when the last caller aborts.
 *
 * On the server (SSR) the wrapper is a no-op pass-through.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function debounce<T extends (...args: any[]) => Promise<any>>(
    cb: T,
    options?: DebounceOptions
): (...args: Parameters<T>) => Promise<Awaited<ReturnType<T>>> {
    const delay = options?.delay ?? DEFAULT_DEBOUNCE_MS;
    const log: NoopLogger = options?.log ? console.log.bind(console) : () => {};

    if (!browser) {
        return cb as (...args: Parameters<T>) => Promise<Awaited<ReturnType<T>>>;
    }

    type V = Awaited<ReturnType<T>>;

    interface Caller {
        signal: AbortSignal | undefined;
        resolve: (v: V) => void;
        reject: (e: unknown) => void;
    }

    interface Entry {
        controller: AbortController;
        callers: Set<Caller>;
        /** True once the timer has fired and `cb` has been invoked. */
        fired: boolean;
    }

    const timers = new Map<string, number>();
    const entries = new Map<string, Entry>();

    return (...args: Parameters<T>) => {
        // Convention: a trailing AbortSignal is the caller's signal.
        const callerSignal =
            args.length > 0 && args[args.length - 1] instanceof AbortSignal
                ? (args[args.length - 1] as AbortSignal)
                : undefined;
        const cbArgs = callerSignal ? args.slice(0, -1) : args;

        if (callerSignal?.aborted) {
            return Promise.reject(callerSignal.reason ?? new DOMException('Aborted', 'AbortError'));
        }

        // The dedupe key ignores the caller's signal so calls with the same
        // non-signal args coalesce regardless of whether a signal was passed.
        const key = JSON.stringify(cbArgs);

        let entry = entries.get(key);
        if (entry === undefined) {
            entry = { controller: new AbortController(), callers: new Set(), fired: false };
            entries.set(key, entry);
            log('[debounce]', key, 'created entry');
        } else {
            log(
                '[debounce]',
                key,
                'reused entry, callers=',
                entry.callers.size,
                'fired=',
                entry.fired,
                'ctrlAborted=',
                entry.controller.signal.aborted
            );
        }

        const promise = attachCaller(key, entry, callerSignal);

        if (!entry.fired) {
            const existingTimer = timers.get(key);
            if (existingTimer !== undefined) {
                log('[debounce]', key, 'reset timer');
                window.clearTimeout(existingTimer);
            }
            const timer = window.setTimeout(() => {
                timers.delete(key);
                // The entry may have been finalized (e.g. all callers
                // aborted) — don't invoke the callback in that case.
                if (entries.get(key) !== entry || entry.callers.size === 0) {
                    log('[debounce]', key, 'timer fired, no callers');
                    return;
                }
                log('[debounce]', key, 'timer fired, calling cb');
                fire(key, entry, cbArgs);
            }, delay);
            timers.set(key, timer);
        } else {
            log('[debounce]', key, 'already in-flight, not scheduling new timer');
        }

        return promise;
    };

    function attachCaller(key: string, entry: Entry, signal: AbortSignal | undefined): Promise<V> {
        return new Promise<V>((resolve, reject) => {
            const caller: Caller = { signal, resolve, reject };
            entry.callers.add(caller);

            if (signal) {
                signal.addEventListener(
                    'abort',
                    () => {
                        // Stale listener — the entry was finalized after
                        // the listener was registered.
                        if (entries.get(key) !== entry) {
                            return;
                        }
                        log('[debounce]', key, 'caller abort, callersBefore=', entry.callers.size);
                        entry.callers.delete(caller);
                        reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
                        if (entry.callers.size === 0) {
                            log('[debounce]', key, 'last caller aborted, finalizing');
                            // Last caller aborted. Abort the underlying
                            // fetch and finalize the entry so the next call
                            // for this key starts a fresh entry rather than
                            // deduping onto this doomed one.
                            entry.controller.abort();
                            entries.delete(key);
                        }
                    },
                    { once: true }
                );
            }
        });
    }

    function fire(key: string, entry: Entry, cbArgs: unknown[]) {
        entry.fired = true;
        const hadSignal = [...entry.callers].some((c) => c.signal !== undefined);
        const callArgs = hadSignal ? [...cbArgs, entry.controller.signal] : cbArgs;
        log(
            '[debounce]',
            key,
            'fire, hadSignal=',
            hadSignal,
            'ctrlAborted=',
            entry.controller.signal.aborted
        );

        const finalize = () => {
            log('[debounce]', key, 'finalize');
            if (entries.get(key) === entry) {
                entries.delete(key);
            }
        };

        // Iterate the live `entry.callers` set at resolve time so callers
        // that joined via dedupe during the in-flight phase also receive
        // the result. Callers that aborted have already been removed from
        // the set; their promises are already settled, so the resolve/reject
        // calls on them are no-ops.
        try {
            (cb as (...a: unknown[]) => Promise<V>)(...callArgs).then(
                (v) => {
                    log('[debounce]', key, 'cb resolved');
                    finalize();
                    for (const c of entry.callers) {
                        c.resolve(v);
                    }
                },
                (e) => {
                    if (entry.callers.size === 0) {
                        // No callers are waiting anymore (they all aborted).
                        // Swallow the rejection rather than logging a spurious
                        // 'cb rejected' for a fetch nobody cares about.
                        log('[debounce]', key, 'cb rejected, no interested callers');
                        finalize();
                        return;
                    }
                    log('[debounce]', key, 'cb rejected', e);
                    finalize();
                    for (const c of entry.callers) {
                        c.reject(e);
                    }
                }
            );
        } catch (e) {
            // `cb` threw synchronously; treat as a rejection.
            finalize();
            for (const c of entry.callers) {
                c.reject(e);
            }
        }
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
