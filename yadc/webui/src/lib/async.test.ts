import { afterEach, describe, expect, it, vi } from 'vitest';
import { debounce } from '$lib/async';

afterEach(() => {
    vi.useRealTimers();
});

/** Advance fake timers and flush the microtask queue. */
async function tick(ms: number): Promise<void> {
    vi.advanceTimersByTime(ms);
    await vi.runAllTimersAsync();
}

/** Create a controllable promise: resolve/reject via returned callbacks. */
function deferred<T = void>(): {
    promise: Promise<T>;
    resolve: (v: T) => void;
    reject: (e: unknown) => void;
} {
    let resolve!: (v: T) => void;
    let reject!: (e: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { promise, resolve, reject };
}

describe('debounce', () => {
    it('calls the callback after the delay', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, 50);

        const promise = debounced('a');
        expect(cb).not.toHaveBeenCalled();

        await tick(50);
        await expect(promise).resolves.toBe('result');
        expect(cb).toHaveBeenCalledOnce();
        expect(cb).toHaveBeenCalledWith('a');
    });

    it('dedupes calls with the same arguments', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, 50);

        const p1 = debounced('a');
        const p2 = debounced('a');

        // Both calls return the same promise (dedupe).
        expect(p1).toBe(p2);

        await tick(50);
        const result = await p1;
        expect(result).toBe('result');
        expect(cb).toHaveBeenCalledOnce();
    });

    it('resets the timer on subsequent calls (true debounce)', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, 50);

        debounced('a');
        vi.advanceTimersByTime(40);

        // Second call resets the timer.
        debounced('a');
        vi.advanceTimersByTime(40);
        expect(cb).not.toHaveBeenCalled();

        await tick(10);
        expect(cb).toHaveBeenCalledOnce();
    });

    it('handles different argument sets independently', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, 50);

        const p1 = debounced('a');
        const p2 = debounced('b');

        await tick(50);
        await Promise.all([p1, p2]);

        expect(cb).toHaveBeenCalledTimes(2);
        expect(cb).toHaveBeenCalledWith('a');
        expect(cb).toHaveBeenCalledWith('b');
    });
});

describe('debounce with AbortSignal', () => {
    it('passes the controller signal to the callback, not the caller signal', async () => {
        vi.useFakeTimers();

        let receivedSignal: AbortSignal | undefined;
        const cb = vi.fn().mockImplementation(async (_arg: string, signal?: AbortSignal) => {
            receivedSignal = signal;
            return 'result';
        });
        const debounced = debounce(cb, 50);

        const caller = new AbortController();
        const promise = debounced('a', caller.signal);

        await tick(50);
        await expect(promise).resolves.toBe('result');

        // The callback receives the debouncer's internal signal, not the caller's.
        expect(receivedSignal).toBeDefined();
        expect(receivedSignal).not.toBe(caller.signal);
        expect(receivedSignal!.aborted).toBe(false);
    });

    it('rejects immediately if the caller signal is already aborted', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, 50);

        const caller = new AbortController();
        caller.abort();

        await expect(debounced('a', caller.signal)).rejects.toThrow('This operation was aborted');

        // Timer fires, but the callback should not have been called since the
        // promise was already rejected.
        vi.advanceTimersByTime(50);
        expect(cb).not.toHaveBeenCalled();
    });

    it('aborts the in-flight call when the caller signal aborts', async () => {
        vi.useFakeTimers();

        const { promise: inflight, resolve: resolveInflight } = deferred<string>();
        let receivedSignal: AbortSignal | undefined;
        const cb = vi.fn().mockImplementation(async (_arg: string, signal?: AbortSignal) => {
            receivedSignal = signal;
            return inflight;
        });
        const debounced = debounce(cb, 50);

        const caller = new AbortController();
        const promise = debounced('a', caller.signal);

        // Timer fires — callback starts executing, blocked on `inflight`.
        vi.advanceTimersByTime(50);
        // Flush microtasks so the cb body runs, but it's still awaiting `inflight`.
        await new Promise<void>((r) => queueMicrotask(r));

        // Abort while the callback is in-flight.
        caller.abort();

        // Resolve the in-flight promise — the callback returns, fire() resolves.
        resolveInflight('result');
        await expect(promise).resolves.toBe('result');

        // The controller's signal (passed to cb) was aborted.
        expect(receivedSignal!.aborted).toBe(true);
    });

    it('aborts if any linked signal aborts (multiple callers)', async () => {
        vi.useFakeTimers();

        const { promise: inflight, resolve: resolveInflight } = deferred<string>();
        let receivedSignal: AbortSignal | undefined;
        const cb = vi.fn().mockImplementation(async (_arg: string, signal?: AbortSignal) => {
            receivedSignal = signal;
            return inflight;
        });
        const debounced = debounce(cb, 50);

        const caller1 = new AbortController();
        const caller2 = new AbortController();

        // First caller starts the timer.
        const p1 = debounced('a', caller1.signal);

        // Second caller resets the timer — same key, deduped promise.
        const p2 = debounced('a', caller2.signal);
        expect(p1).toBe(p2);

        // Timer fires.
        vi.advanceTimersByTime(50);
        await new Promise<void>((r) => queueMicrotask(r));

        // Aborting caller2's signal should abort the entry's controller.
        caller2.abort();
        expect(receivedSignal!.aborted).toBe(true);

        resolveInflight('result');
        await expect(p1).resolves.toBe('result');
    });

    it('does not pass a signal when caller does not provide one', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, 50);

        const promise = debounced('a');
        await tick(50);
        await expect(promise).resolves.toBe('result');

        // Callback called with just the arg, no signal appended.
        expect(cb).toHaveBeenCalledWith('a');
    });

    it('cleans up listeners after the callback resolves', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, 50);

        const caller = new AbortController();
        const promise = debounced('a', caller.signal);

        await tick(50);
        await expect(promise).resolves.toBe('result');

        // After resolution, aborting the caller's signal should be a no-op
        // (no uncaught errors, no lingering listeners).
        caller.abort();
        // Give microtasks a chance to surface any stray errors.
        await tick(0);
    });

    it('allows a fresh call after the previous call resolves', async () => {
        vi.useFakeTimers();

        let callCount = 0;
        const cb = vi.fn().mockImplementation(async () => {
            callCount++;
            return `result-${callCount}`;
        });
        const debounced = debounce(cb, 50);

        const caller1 = new AbortController();
        const p1 = debounced('a', caller1.signal);
        await tick(50);
        await expect(p1).resolves.toBe('result-1');

        // New call with same args should start fresh (entry was finalized).
        const caller2 = new AbortController();
        const p2 = debounced('a', caller2.signal);
        await tick(50);
        await expect(p2).resolves.toBe('result-2');

        expect(cb).toHaveBeenCalledTimes(2);
    });
});
