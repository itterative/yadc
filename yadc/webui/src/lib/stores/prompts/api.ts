/**
 * POST /api/prompts/generate — NDJSON streaming.
 *
 * The backend returns one JSON object per line:
 *   {"type": "token", "text": "...", "reasoning": "..."}
 *   {"type": "done"}
 *   {"type": "error", "message": "..."}
 *
 * We POST the request, then iterate the response body as a stream of
 * ``StreamEvent``s and dispatch each via the supplied callbacks. The
 * caller controls cancellation through the ``AbortSignal`` —
 * aborting the signal aborts the in-flight fetch and the
 * ReadableStream reader.
 *
 * Preflight (HTTP 400/403) errors propagate as a regular thrown
 * ``Error`` (the same convention used by the other API helpers);
 * mid-stream ``{"type": "error"}`` lines surface via
 * ``callbacks.onError`` so the stream can still be torn down
 * cleanly via the abort signal.
 */

import { API_BASE, apiErrorMessage } from '$lib/api';
import type { PromptGenRequest, StreamEvent } from './types';

export interface StreamCallbacks {
    onToken?: (text: string) => void;
    onReasoning?: (text: string) => void;
    onReasoningSummary?: (text: string) => void;
    onDone?: () => void;
    onError?: (message: string) => void;
}

/** Fetch an image from a dataset and return it as a base64 data URL.
 *
 *  Used by the "Add from dataset" example flow — the frontend encodes
 *  the image locally so the backend stays stateless about image
 *  sources. ``mime`` defaults to ``image/jpeg`` for the common case;
 *  pass the actual MIME type from a fetch response when known. */
export async function fetchImageAsDataUrl(
    datasetName: string,
    imageId: number,
    signal?: AbortSignal
): Promise<{ dataUrl: string; mime: string }> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/media`,
        { signal }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    const blob = await res.blob();
    const mime = blob.type || 'image/jpeg';
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = () => reject(reader.error ?? new Error('FileReader failed'));
        reader.onload = () => {
            const dataUrl = reader.result;
            if (typeof dataUrl !== 'string') {
                reject(new Error('FileReader did not return a string'));
                return;
            }
            resolve({ dataUrl, mime });
        };
        reader.readAsDataURL(blob);
    });
}

/** Read a local ``File`` as a base64 data URL. */
export function readFileAsDataUrl(file: File): Promise<{ dataUrl: string; mime: string }> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = () => reject(reader.error ?? new Error('FileReader failed'));
        reader.onload = () => {
            const dataUrl = reader.result;
            if (typeof dataUrl !== 'string') {
                reject(new Error('FileReader did not return a string'));
                return;
            }
            resolve({ dataUrl, mime: file.type || 'image/jpeg' });
        };
        reader.readAsDataURL(file);
    });
}

/** POST the generate request and dispatch streaming events to ``callbacks``. */
export async function generatePrompt(
    request: PromptGenRequest,
    callbacks: StreamCallbacks,
    signal?: AbortSignal
): Promise<void> {
    const res = await fetch(`${API_BASE}/api/prompts/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal
    });

    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    if (!res.body) {
        throw new Error('Generation response has no body');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            // Split on newlines; keep the trailing partial line in the
            // buffer for the next chunk.
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';

            for (const line of lines) {
                if (!line.trim()) {
                    continue;
                }
                let event: StreamEvent;
                try {
                    event = JSON.parse(line) as StreamEvent;
                } catch {
                    // Skip malformed lines — the backend should never
                    // produce them, but a corrupt chunk shouldn't
                    // crash the whole generation.
                    continue;
                }
                dispatch(event, callbacks);

                // A "done" or "error" line ends the stream — stop
                // reading further chunks.
                if (event.type === 'done' || event.type === 'error') {
                    return;
                }
            }
        }

        // Flush any remaining buffer content. The backend may close
        // the response without a trailing newline on the last event,
        // which leaves the final line in the buffer. Also run the
        // decoder in non-streaming mode to drain any pending bytes.
        buffer += decoder.decode();
        if (buffer.trim()) {
            try {
                const event = JSON.parse(buffer) as StreamEvent;
                dispatch(event, callbacks);
            } catch {
                /* malformed trailing line — ignore */
            }
        }

        // Stream ended without an explicit "done" event. Treat it as
        // natural completion (backend may close the response on EOF).
        callbacks.onDone?.();
    } finally {
        // Release the reader. If the caller aborted, the reader is
        // already in a closed state; ``cancel()`` is a no-op then.
        try {
            await reader.cancel();
        } catch {
            /* already closed */
        }
    }
}

function dispatch(event: StreamEvent, callbacks: StreamCallbacks): void {
    switch (event.type) {
        case 'token':
            if (event.text) {
                callbacks.onToken?.(event.text);
            }
            if (event.reasoning) {
                callbacks.onReasoning?.(event.reasoning);
            }
            if (event.reasoning_summary) {
                callbacks.onReasoningSummary?.(event.reasoning_summary);
            }
            break;
        case 'done':
            callbacks.onDone?.();
            break;
        case 'error':
            callbacks.onError?.(event.message);
            break;
    }
}
