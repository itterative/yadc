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

    const timers = new Map<string, number>();
    const entries = new Map<
        string,
        { resolve: (v: V) => void; reject: (e: unknown) => void; promise: Promise<V> }
    >();

    return (...args: Parameters<T>) => {
        const key = JSON.stringify(args);

        // Cancel and restart the timer (true debounce — reset on each call).
        const existingTimer = timers.get(key);
        if (existingTimer !== undefined) {
            window.clearTimeout(existingTimer);
        }

        // Dedupe: reuse the existing promise if one is already pending or in-flight.
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
            entry = { resolve, reject, promise };
            entries.set(key, entry);
        }

        // Start / restart the debounce timer.
        const timer = window.setTimeout(() => {
            timers.delete(key);
            fire(key, args);
        }, delay);
        timers.set(key, timer);

        return entry.promise;
    };

    function fire(key: string, args: Parameters<T>) {
        const entry = entries.get(key);
        if (!entry) {
            return;
        }
        try {
            cb(...args)
                .then((v) => {
                    entries.delete(key);
                    entry.resolve(v);
                })
                .catch((e) => {
                    entries.delete(key);
                    entry.reject(e);
                });
        } catch (e) {
            entries.delete(key);
            entry.reject(e);
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
