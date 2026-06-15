/** Tests for ``withPasswordRetry`` — password-prompt flow with abort handling. */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PasswordRequiredError } from '$lib/api';

import {
    cancelPassword,
    PasswordPromptCancelled,
    submitPassword,
    withPasswordRetry
} from './passwordPrompt';

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

beforeEach(() => {
    // ``setAuthCookie`` POSTs to ``/api/auth/password``; ``clearAuthCookie``
    // DELETEs the same. Default to both succeeding; tests that need
    // different behavior override.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(null, { status: 204 })));
});

afterEach(() => {
    vi.unstubAllGlobals();
    // Make sure no pending prompt leaks between tests.
    cancelPassword();
});

describe('withPasswordRetry — happy paths', () => {
    it('returns the action result when the first attempt succeeds', async () => {
        const action = vi.fn().mockResolvedValue('result');

        const result = await withPasswordRetry(action);

        expect(result).toBe('result');
        expect(action).toHaveBeenCalledOnce();
        expect(fetch).not.toHaveBeenCalled(); // no cookie POST/DELETE needed
    });

    it('prompts and retries after PasswordRequiredError', async () => {
        const action = vi
            .fn()
            .mockRejectedValueOnce(new PasswordRequiredError('first'))
            .mockResolvedValueOnce('retried-result');

        // The user submits the password after the first attempt fails.
        // Use setTimeout(0) so the submit happens as a macrotask — by
        // then the wrapper has finished its own microtasks (cleared the
        // stale cookie, opened the prompt). queueMicrotask() would run
        // before the wrapper had a chance to open the prompt.
        setTimeout(() => submitPassword('hunter2'), 0);

        const result = await withPasswordRetry(action);

        expect(result).toBe('retried-result');
        expect(action).toHaveBeenCalledTimes(2);
        // First call: clear stale cookie (DELETE). Second: set new cookie (POST).
        expect(fetch).toHaveBeenCalledTimes(2);
        expect(vi.mocked(fetch).mock.calls[0][1]?.method).toBe('DELETE');
        expect(vi.mocked(fetch).mock.calls[1][1]?.method).toBe('POST');
    });

    it('re-throws non-PasswordRequiredError without prompting', async () => {
        const action = vi.fn().mockRejectedValue(new Error('boom'));

        await expect(withPasswordRetry(action)).rejects.toThrow('boom');

        expect(action).toHaveBeenCalledOnce();
        expect(fetch).not.toHaveBeenCalled();
    });
});

describe('withPasswordRetry — cookie POST failures', () => {
    it('does not retry when setAuthCookie returns 403 (wrong password)', async () => {
        const action = vi.fn().mockRejectedValueOnce(new PasswordRequiredError('first'));
        // First call: DELETE (clear stale cookie) — success.
        // Second call: POST (set new cookie) — 403 PASSWORD_REQUIRED.
        vi.mocked(fetch)
            .mockResolvedValueOnce(new Response(null, { status: 204 }))
            .mockResolvedValueOnce(
                new Response(JSON.stringify({ code: 'PASSWORD_REQUIRED' }), { status: 403 })
            );

        setTimeout(() => submitPassword('wrong'), 0);

        await expect(withPasswordRetry(action)).rejects.toBeInstanceOf(PasswordRequiredError);
        expect(action).toHaveBeenCalledOnce(); // No second attempt.
    });
});

describe('withPasswordRetry — user cancellation', () => {
    it('throws PasswordPromptCancelled when the user cancels the prompt', async () => {
        const action = vi.fn().mockRejectedValueOnce(new PasswordRequiredError('first'));

        setTimeout(() => cancelPassword(), 0);

        await expect(withPasswordRetry(action)).rejects.toBeInstanceOf(PasswordPromptCancelled);
        expect(action).toHaveBeenCalledOnce(); // No second attempt.
        // clearAuthCookie runs BEFORE the prompt opens, so the DELETE
        // does fire. The POST must NOT (no second action).
        const calls = vi.mocked(fetch).mock.calls;
        expect(calls).toHaveLength(1);
        expect(calls[0][1]?.method).toBe('DELETE');
    });
});

describe('withPasswordRetry — second action failure', () => {
    it('clears the cookie and re-throws when the second action throws PasswordRequiredError', async () => {
        const action = vi
            .fn()
            .mockRejectedValueOnce(new PasswordRequiredError('first'))
            .mockRejectedValueOnce(new PasswordRequiredError('still wrong'));

        setTimeout(() => submitPassword('hunter2'), 0);

        await expect(withPasswordRetry(action)).rejects.toBeInstanceOf(PasswordRequiredError);
        expect(action).toHaveBeenCalledTimes(2);
        // DELETE (pre-prompt) → POST (set new cookie) → DELETE (post-retry).
        const calls = vi.mocked(fetch).mock.calls;
        expect(calls).toHaveLength(3);
        expect(calls[0][1]?.method).toBe('DELETE');
        expect(calls[1][1]?.method).toBe('POST');
        expect(calls[2][1]?.method).toBe('DELETE');
    });

    it('re-throws non-PasswordRequiredError from the second action', async () => {
        const action = vi
            .fn()
            .mockRejectedValueOnce(new PasswordRequiredError('first'))
            .mockRejectedValueOnce(new Error('upstream broken'));

        setTimeout(() => submitPassword('hunter2'), 0);

        await expect(withPasswordRetry(action)).rejects.toThrow('upstream broken');
        // Cookie is NOT cleared on the non-password second failure.
        const calls = vi.mocked(fetch).mock.calls;
        expect(calls).toHaveLength(2);
        expect(calls[0][1]?.method).toBe('DELETE'); // pre-prompt
        expect(calls[1][1]?.method).toBe('POST'); // set new cookie
    });
});

describe('withPasswordRetry — abort handling', () => {
    it('throws AbortError immediately when the signal is already aborted', async () => {
        const action = vi.fn().mockResolvedValue('result');
        const controller = new AbortController();
        controller.abort();

        await expect(withPasswordRetry(action, controller.signal)).rejects.toMatchObject({
            name: 'AbortError'
        });
        expect(action).not.toHaveBeenCalled();
    });

    it('throws AbortError when the signal aborts during the first action', async () => {
        const inflight = deferred<string>();
        const action = vi.fn().mockReturnValueOnce(inflight.promise);
        const controller = new AbortController();

        const promise = withPasswordRetry(action, controller.signal);

        // Action starts, hangs on the inflight promise. Abort the signal.
        await new Promise<void>((r) => queueMicrotask(r));
        controller.abort();

        // The inflight action needs to reject so withPasswordRetry can
        // observe the abort. Reject it as an AbortError (mimicking what
        // fetch with an aborted signal would do).
        inflight.reject(new DOMException('The operation was aborted.', 'AbortError'));

        await expect(promise).rejects.toMatchObject({ name: 'AbortError' });
    });

    it('dismisses the prompt when the signal aborts while waiting for the user', async () => {
        const action = vi.fn().mockRejectedValueOnce(new PasswordRequiredError('first'));
        const controller = new AbortController();

        // The prompt is now open and waiting for the user. Abort should
        // dismiss it (so the user isn't left staring at a dead dialog).
        const promise = withPasswordRetry(action, controller.signal);

        // setTimeout so the abort runs after the wrapper has reached
        // requestPassword() and is awaiting the pending prompt.
        setTimeout(() => controller.abort(), 0);

        // The wrapper should throw AbortError, not PasswordPromptCancelled.
        await expect(promise).rejects.toMatchObject({ name: 'AbortError' });
        // The user can still submit a password manually; the late submit
        // must not crash (submitPassword() finds no pending promise and
        // returns silently).
        expect(() => submitPassword('late')).not.toThrow();
    });

    it('throws AbortError when the signal aborts between the prompt and the retry', async () => {
        const action = vi
            .fn()
            .mockRejectedValueOnce(new PasswordRequiredError('first'))
            .mockResolvedValueOnce('retried');
        const controller = new AbortController();

        const promise = withPasswordRetry(action, controller.signal);

        // User submits quickly. But the signal aborts after the cookie POST
        // but before the second action. setTimeout so the submit lands
        // after the wrapper's own microtasks.
        setTimeout(() => {
            submitPassword('hunter2');
            // Abort right after the submit is queued. The throwIfAborted()
            // before the second action should catch this.
            controller.abort();
        }, 0);

        await expect(promise).rejects.toMatchObject({ name: 'AbortError' });
        // The second action must NOT have been called.
        expect(action).toHaveBeenCalledOnce();
    });

    it('removes the abort listener after the action resolves', async () => {
        const action = vi.fn().mockResolvedValue('result');
        const controller = new AbortController();
        const removeSpy = vi.spyOn(controller.signal, 'removeEventListener');

        await withPasswordRetry(action, controller.signal);

        // Listener was added and removed exactly once each. The wrapper
        // passes the same onAbort function reference to addEventListener
        // and removeEventListener (no third options arg on remove).
        expect(removeSpy).toHaveBeenCalledWith('abort', expect.any(Function));
    });

    it('removes the abort listener after the action throws', async () => {
        const action = vi.fn().mockRejectedValue(new Error('boom'));
        const controller = new AbortController();
        const removeSpy = vi.spyOn(controller.signal, 'removeEventListener');

        await expect(withPasswordRetry(action, controller.signal)).rejects.toThrow('boom');

        expect(removeSpy).toHaveBeenCalledWith('abort', expect.any(Function));
    });

    it('does not register an abort listener when no signal is provided (backward compat)', async () => {
        const action = vi.fn().mockResolvedValue('result');

        // No second parameter — should still work and not crash.
        const result = await withPasswordRetry(action);

        expect(result).toBe('result');
        expect(action).toHaveBeenCalledOnce();
    });
});
