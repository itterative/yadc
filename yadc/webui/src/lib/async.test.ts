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
        const debounced = debounce(cb, { delay: 50 });

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
        const debounced = debounce(cb, { delay: 50 });

        const p1 = debounced('a');
        const p2 = debounced('a');

        // Both calls are backed by the same underlying fetch, but each caller
        // gets their own promise so aborts are isolated.
        expect(p1).not.toBe(p2);

        await tick(50);
        const result = await p1;
        expect(result).toBe('result');
        expect(await p2).toBe('result');
        expect(cb).toHaveBeenCalledOnce();
    });

    it('resets the timer on subsequent calls (true debounce)', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, { delay: 50 });

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
        const debounced = debounce(cb, { delay: 50 });

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
        const debounced = debounce(cb, { delay: 50 });

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
        const debounced = debounce(cb, { delay: 50 });

        const caller = new AbortController();
        caller.abort();

        await expect(debounced('a', caller.signal)).rejects.toThrow('This operation was aborted');

        // Timer fires, but the callback should not have been called since the
        // promise was already rejected.
        vi.advanceTimersByTime(50);
        expect(cb).not.toHaveBeenCalled();
    });

    it('rejects only the aborted caller when a single caller aborts', async () => {
        vi.useFakeTimers();

        const { promise: inflight, resolve: resolveInflight } = deferred<string>();
        let receivedSignal: AbortSignal | undefined;
        const cb = vi.fn().mockImplementation(async (_arg: string, signal?: AbortSignal) => {
            receivedSignal = signal;
            return inflight;
        });
        const debounced = debounce(cb, { delay: 50 });

        const caller = new AbortController();
        const promise = debounced('a', caller.signal);

        // Timer fires — callback starts executing, blocked on `inflight`.
        vi.advanceTimersByTime(50);
        // Flush microtasks so the cb body runs, but it's still awaiting `inflight`.
        await new Promise<void>((r) => queueMicrotask(r));

        // Abort while the callback is in-flight.
        caller.abort();

        // The aborted caller's promise rejects.
        await expect(promise).rejects.toThrow('This operation was aborted');

        // The controller's signal (passed to cb) was aborted.
        expect(receivedSignal!.aborted).toBe(true);

        // Resolving the in-flight promise should not cause any uncaught rejection.
        resolveInflight('result');
        await tick(0);
    });

    it('continues the fetch for remaining callers when one aborts', async () => {
        vi.useFakeTimers();

        const { promise: inflight, resolve: resolveInflight } = deferred<string>();
        let receivedSignal: AbortSignal | undefined;
        const cb = vi.fn().mockImplementation(async (_arg: string, signal?: AbortSignal) => {
            receivedSignal = signal;
            return inflight;
        });
        const debounced = debounce(cb, { delay: 50 });

        const caller1 = new AbortController();
        const caller2 = new AbortController();

        // First caller starts the timer.
        const p1 = debounced('a', caller1.signal);

        // Second caller dedupes onto the same entry.
        const p2 = debounced('a', caller2.signal);
        expect(p1).not.toBe(p2);

        // Timer fires.
        vi.advanceTimersByTime(50);
        await new Promise<void>((r) => queueMicrotask(r));

        // Aborting caller2's signal should NOT abort the entry's controller
        // because caller1 is still interested.
        caller2.abort();
        expect(receivedSignal!.aborted).toBe(false);

        // caller2's promise rejects.
        await expect(p2).rejects.toThrow('This operation was aborted');

        // caller1's promise resolves normally when the fetch completes.
        resolveInflight('result');
        await expect(p1).resolves.toBe('result');
    });

    it('passes a signal when any deduped caller provided one', async () => {
        vi.useFakeTimers();

        let receivedSignal: AbortSignal | undefined;
        const cb = vi.fn().mockImplementation(async (_arg: string, signal?: AbortSignal) => {
            receivedSignal = signal;
            return 'result';
        });
        const debounced = debounce(cb, { delay: 50 });

        const caller = new AbortController();
        const p1 = debounced('a', caller.signal);

        // Second caller does not pass a signal, but the first one did.
        const p2 = debounced('a');
        expect(p1).not.toBe(p2);

        await tick(50);
        await expect(p1).resolves.toBe('result');
        await expect(p2).resolves.toBe('result');

        // Because at least one deduped caller passed a signal, the callback
        // receives the debouncer's internal controller signal.
        expect(receivedSignal).toBeDefined();
        expect(receivedSignal).not.toBe(caller.signal);
    });

    it('does not pass a signal when caller does not provide one', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, { delay: 50 });

        const promise = debounced('a');
        await tick(50);
        await expect(promise).resolves.toBe('result');

        // Callback called with just the arg, no signal appended.
        expect(cb).toHaveBeenCalledWith('a');
    });

    it('cleans up listeners after the callback resolves', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, { delay: 50 });

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
        const debounced = debounce(cb, { delay: 50 });

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

    it('does not cancel the fetch when the only caller aborts before the timer fires', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, { delay: 50 });

        const caller = new AbortController();
        const promise = debounced('a', caller.signal);

        // Abort during the debounce window, before the callback is invoked.
        caller.abort();

        await expect(promise).rejects.toThrow('This operation was aborted');

        // The timer should still fire, but with no interested callers the
        // callback is not invoked.
        vi.advanceTimersByTime(50);
        expect(cb).not.toHaveBeenCalled();
    });

    it('keeps the fetch alive for remaining callers when one aborts during the debounce window', async () => {
        vi.useFakeTimers();

        const cb = vi.fn().mockResolvedValue('result');
        const debounced = debounce(cb, { delay: 50 });

        const caller1 = new AbortController();
        const caller2 = new AbortController();

        const p1 = debounced('a', caller1.signal);
        const p2 = debounced('a', caller2.signal);

        // Abort the first caller during the debounce window.
        caller1.abort();

        // First caller's promise rejects.
        await expect(p1).rejects.toThrow('This operation was aborted');

        // Second caller still gets the result when the timer fires.
        await tick(50);
        await expect(p2).resolves.toBe('result');
        expect(cb).toHaveBeenCalledOnce();
    });

    it('keeps the fetch alive for remaining callers when one aborts in-flight', async () => {
        vi.useFakeTimers();

        const { promise: inflight, resolve: resolveInflight } = deferred<string>();
        let receivedSignal: AbortSignal | undefined;
        const cb = vi.fn().mockImplementation(async (_arg: string, signal?: AbortSignal) => {
            receivedSignal = signal;
            return inflight;
        });
        const debounced = debounce(cb, { delay: 50 });

        const caller1 = new AbortController();
        const caller2 = new AbortController();

        const p1 = debounced('a', caller1.signal);
        const p2 = debounced('a', caller2.signal);

        // Timer fires and fetch starts.
        vi.advanceTimersByTime(50);
        await new Promise<void>((r) => queueMicrotask(r));

        // Abort the first caller while in-flight.
        caller1.abort();

        // First caller's promise rejects, but the controller is not aborted
        // because caller2 is still interested.
        await expect(p1).rejects.toThrow('This operation was aborted');
        expect(receivedSignal!.aborted).toBe(false);

        // Completing the fetch resolves the remaining caller.
        resolveInflight('result');
        await expect(p2).resolves.toBe('result');
    });

    it('gives a fresh controller to a new caller after all prior callers aborted in-flight', async () => {
        vi.useFakeTimers();

        const { promise: inflight, resolve: resolveInflight } = deferred<string>();
        let receivedSignal: AbortSignal | undefined;
        const cb = vi.fn().mockImplementation(async (_arg: string, signal?: AbortSignal) => {
            receivedSignal = signal;
            return inflight;
        });
        const debounced = debounce(cb, { delay: 50 });

        const caller1 = new AbortController();
        const p1 = debounced('a', caller1.signal);

        // Timer fires and fetch starts.
        vi.advanceTimersByTime(50);
        await new Promise<void>((r) => queueMicrotask(r));
        expect(cb).toHaveBeenCalledOnce();

        // The only caller aborts while in-flight. This finalizes the entry.
        caller1.abort();
        await expect(p1).rejects.toThrow('This operation was aborted');

        // A new caller arrives before the old fetch resolves. It must get a
        // brand-new entry with a fresh (non-aborted) controller.
        const caller2 = new AbortController();
        const p2 = debounced('a', caller2.signal);

        // New timer should fire and start a second fetch.
        vi.advanceTimersByTime(50);
        await new Promise<void>((r) => queueMicrotask(r));

        expect(cb).toHaveBeenCalledTimes(2);
        expect(receivedSignal!.aborted).toBe(false);

        resolveInflight('result');
        await expect(p2).resolves.toBe('result');
    });

    it('does not let an old aborted fetch clobber a new entry for the same key', async () => {
        // Matches the user's log flow:
        //   1. entry created, timer fires, cb in-flight
        //   2. only caller aborts -> entry finalized, controller aborted
        //   3. new caller arrives with same key -> new entry created
        //   4. old fetch rejects with AbortError
        //   5. new caller must still get a fresh fetch, not be orphaned.
        vi.useFakeTimers();

        const inflights: ReturnType<typeof deferred<string>>[] = [];
        const cb = vi.fn().mockImplementation(async (_arg: string, signal?: AbortSignal) => {
            const d = deferred<string>();
            signal?.addEventListener(
                'abort',
                () => {
                    d.reject(new DOMException('The operation was aborted.', 'AbortError'));
                },
                { once: true }
            );
            inflights.push(d);
            return d.promise;
        });
        const debounced = debounce(cb, { delay: 50 });

        const caller1 = new AbortController();
        const p1 = debounced('a', caller1.signal);

        // Timer fires and fetch starts.
        await vi.advanceTimersByTimeAsync(50);
        expect(cb).toHaveBeenCalledOnce();
        expect(inflights).toHaveLength(1);

        // Caller aborts while in-flight. This also rejects the underlying
        // fetch, but we do NOT await p1 yet — we want the new caller to arrive
        // before the old fetch's rejection microtask runs.
        caller1.abort();

        // New caller arrives before the old fetch rejection has been processed.
        const caller2 = new AbortController();
        const p2 = debounced('a', caller2.signal);

        // Flush microtasks so the old rejection is processed BEFORE the new
        // debounce timer (a macrotask) has a chance to fire.
        await expect(p1).rejects.toThrow('This operation was aborted');

        // The new caller must not be rejected by the old fetch's AbortError.
        // Its timer must still fire and start a brand-new fetch.
        await vi.advanceTimersByTimeAsync(50);

        expect(cb).toHaveBeenCalledTimes(2);
        expect(inflights).toHaveLength(2);

        // Resolving the second fetch should satisfy the new caller.
        inflights[1].resolve('result');
        await expect(p2).resolves.toBe('result');
    });

    it('does not schedule a second fetch when a new caller dedupes onto an in-flight entry', async () => {
        vi.useFakeTimers();

        const { promise: inflight, resolve: resolveInflight } = deferred<string>();
        const cb = vi.fn().mockImplementation(async () => inflight);
        const debounced = debounce(cb, { delay: 50 });

        const caller1 = new AbortController();
        const p1 = debounced('a', caller1.signal);

        // Timer fires and fetch starts.
        vi.advanceTimersByTime(50);
        await new Promise<void>((r) => queueMicrotask(r));
        expect(cb).toHaveBeenCalledOnce();

        // A second caller arrives while the fetch is still in-flight.
        const caller2 = new AbortController();
        const p2 = debounced('a', caller2.signal);

        // Advancing timers further should NOT trigger another fetch.
        vi.advanceTimersByTime(50);
        await new Promise<void>((r) => queueMicrotask(r));
        expect(cb).toHaveBeenCalledOnce();

        // Both callers share the result of the single fetch.
        resolveInflight('result');
        await expect(p1).resolves.toBe('result');
        await expect(p2).resolves.toBe('result');
    });
});
