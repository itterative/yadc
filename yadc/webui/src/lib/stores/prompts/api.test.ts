/** Tests for the NDJSON streaming reader in ``./api`` and the local-file helper. */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { API_BASE } from '$lib/api';
import { generatePrompt, readFileAsDataUrl } from './api';
import type { PromptGenRequest, StreamEvent } from './types';

function _line(event: StreamEvent): string {
    return JSON.stringify(event);
}

function _ndjson(...events: StreamEvent[]): string {
    return events.map(_line).join('\n') + '\n';
}

/** Build a Response from a string body using a ReadableStream.
 *
 *  ``new Response(blob)`` doesn't expose ``.body`` in jsdom in a
 *  way our stream reader can iterate, so we build a real
 *  ReadableStream that the production code path actually consumes. */
function _streamResponse(body: string, options: { ok?: boolean; status?: number } = {}): Response {
    const ok = options.ok ?? true;
    const status = options.status ?? 200;
    const encoded = new TextEncoder().encode(body);
    const stream = new ReadableStream<Uint8Array>({
        start(controller) {
            controller.enqueue(encoded);
            controller.close();
        }
    });
    return new Response(ok ? stream : null, {
        status,
        statusText: 'OK',
        headers: { 'Content-Type': 'application/x-ndjson' }
    });
}

/** Build a Response from multiple chunks (simulates a chunked server). */
function _chunkedResponse(
    chunks: string[],
    options: { ok?: boolean; status?: number } = {}
): Response {
    const ok = options.ok ?? true;
    const status = options.status ?? 200;
    const encoded = chunks.map((c) => new TextEncoder().encode(c));
    const stream = new ReadableStream<Uint8Array>({
        start(controller) {
            for (const chunk of encoded) {
                controller.enqueue(chunk);
            }
            controller.close();
        }
    });
    return new Response(ok ? stream : null, {
        status,
        statusText: 'OK',
        headers: { 'Content-Type': 'application/x-ndjson' }
    });
}

const _request: PromptGenRequest = {
    env: 'default',
    intent: 'caption cats',
    examples: [],
    focus: 'both'
};

beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
    vi.unstubAllGlobals();
});

describe('generatePrompt — happy path', () => {
    it('POSTs the request to /api/prompts/generate and dispatches token events', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
            _streamResponse(
                _ndjson({ type: 'token', text: 'hello ' }, { type: 'token', text: 'world' })
            )
        );

        const onToken = vi.fn();
        const onDone = vi.fn();
        await generatePrompt(_request, { onToken, onDone });

        expect(fetch).toHaveBeenCalledWith(
            `${API_BASE}/api/prompts/generate`,
            expect.objectContaining({
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(_request)
            })
        );
        expect(onToken).toHaveBeenCalledTimes(2);
        expect(onToken).toHaveBeenNthCalledWith(1, 'hello ');
        expect(onToken).toHaveBeenNthCalledWith(2, 'world');
    });

    it('dispatches reasoning and reasoning_summary separately', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
            _streamResponse(
                _ndjson({
                    type: 'token',
                    text: 'visible',
                    reasoning: 'thinking',
                    reasoning_summary: 'sum'
                })
            )
        );

        const onToken = vi.fn();
        const onReasoning = vi.fn();
        const onReasoningSummary = vi.fn();
        await generatePrompt(_request, { onToken, onReasoning, onReasoningSummary });

        expect(onToken).toHaveBeenCalledWith('visible');
        expect(onReasoning).toHaveBeenCalledWith('thinking');
        expect(onReasoningSummary).toHaveBeenCalledWith('sum');
    });

    it('invokes onDone on a "done" line', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(_streamResponse(_ndjson({ type: 'done' })));

        const onDone = vi.fn();
        await generatePrompt(_request, { onDone });

        expect(onDone).toHaveBeenCalledOnce();
    });

    it('invokes onDone on natural EOF (no explicit done line)', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
            _streamResponse(_ndjson({ type: 'token', text: 'x' }))
        );

        const onDone = vi.fn();
        await generatePrompt(_request, { onDone });

        expect(onDone).toHaveBeenCalledOnce();
    });

    it('invokes onError on a mid-stream error line', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
            _streamResponse(
                _ndjson({ type: 'token', text: 'a' }, { type: 'error', message: 'boom' })
            )
        );

        const onToken = vi.fn();
        const onError = vi.fn();
        const onDone = vi.fn();
        await generatePrompt(_request, { onToken, onError, onDone });

        expect(onToken).toHaveBeenCalledWith('a');
        expect(onError).toHaveBeenCalledWith('boom');
        // ``done`` should NOT be called on an error.
        expect(onDone).not.toHaveBeenCalled();
    });

    it('skips malformed JSON lines without crashing', async () => {
        const body =
            _ndjson({ type: 'token', text: 'a' }) +
            'not-json\n' +
            _ndjson({ type: 'token', text: 'b' });
        vi.mocked(fetch).mockResolvedValueOnce(_streamResponse(body));

        const onToken = vi.fn();
        await generatePrompt(_request, { onToken });

        expect(onToken).toHaveBeenCalledTimes(2);
    });
});

describe('generatePrompt — chunked streaming', () => {
    it('handles a token split across two chunks', async () => {
        // The line is split mid-JSON — the reader must buffer and
        // parse on the next chunk.
        const full = JSON.stringify({ type: 'token', text: 'hello world' });
        vi.mocked(fetch).mockResolvedValueOnce(
            _chunkedResponse([full.slice(0, 15), full.slice(15) + '\n'])
        );

        const onToken = vi.fn();
        await generatePrompt(_request, { onToken });

        expect(onToken).toHaveBeenCalledWith('hello world');
    });

    it('handles a final line with no trailing newline', async () => {
        // No trailing "\n" — EOF should still trigger parsing.
        const body = JSON.stringify({ type: 'token', text: 'x' }); // no newline
        vi.mocked(fetch).mockResolvedValueOnce(_streamResponse(body));

        const onToken = vi.fn();
        await generatePrompt(_request, { onToken });

        expect(onToken).toHaveBeenCalledWith('x');
    });
});

describe('generatePrompt — errors', () => {
    it('throws on HTTP error status (preflight)', async () => {
        vi.mocked(fetch).mockResolvedValueOnce(
            new Response('{"error":"bad request"}', { status: 400 })
        );

        await expect(generatePrompt(_request, {})).rejects.toThrow();
    });

    it('throws when the response has no body', async () => {
        // Use an explicit null body — ``new Response('', ...)`` still
        // exposes a body in jsdom.
        vi.mocked(fetch).mockResolvedValueOnce(new Response(null, { status: 200 }));

        await expect(generatePrompt(_request, {})).rejects.toThrow(/no body/);
    });
});

describe('generatePrompt — abort', () => {
    it('aborts the fetch when the signal fires', async () => {
        // Make the fetch honor the abort signal (mimics real fetch).
        vi.mocked(fetch).mockImplementation((_url, init) => {
            const signal = (init as RequestInit | undefined)?.signal;
            return new Promise<Response>((_, reject) => {
                signal?.addEventListener('abort', () => {
                    reject(new DOMException('Aborted', 'AbortError'));
                });
            });
        });

        const controller = new AbortController();
        const promise = generatePrompt(_request, {}, controller.signal);
        controller.abort();

        await expect(promise).rejects.toMatchObject({ name: 'AbortError' });
    });
});

describe('readFileAsDataUrl', () => {
    it('reads a File and returns a base64 data URL', async () => {
        const file = new File(['hello'], 'cat.png', { type: 'image/png' });
        const { dataUrl, mime } = await readFileAsDataUrl(file);

        expect(mime).toBe('image/png');
        expect(dataUrl).toMatch(/^data:image\/png;base64,/);
    });

    it('falls back to image/jpeg when the File has no type', async () => {
        const file = new File(['hello'], 'cat');
        // ``new File`` defaults ``type`` to '' on jsdom.
        const { mime } = await readFileAsDataUrl(file);
        expect(mime).toBe('image/jpeg');
    });
});
