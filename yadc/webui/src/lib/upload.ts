import { API_BASE } from './api';

export interface UploadProgress {
    loaded: number;
    total: number;
}

export interface UploadOptions {
    url: string;
    method?: string;
    body?: XMLHttpRequestBodyInit | null;
    headers?: Record<string, string>;
    signal?: AbortSignal;
    /** Upload progress (bytes sent to the server). */
    onProgress?: (progress: UploadProgress) => void;
    /** Streaming response — called for each complete line received so far.
     *  Only complete lines (terminated by `\n`) are yielded. Use this for
     *  NDJSON streaming endpoints where the server flushes one JSON object per line. */
    onChunk?: (line: string) => void;
}

class UploadResponse {
    ok: boolean;
    status: number;
    statusText: string;
    url: string;

    private _bodyText: string;

    constructor(status: number, statusText: string, url: string, bodyText: string) {
        this.status = status;
        this.statusText = statusText;
        this.url = url;
        this.ok = status >= 200 && status < 300;
        this._bodyText = bodyText;
    }

    async text(): Promise<string> {
        return this._bodyText;
    }

    async json(): Promise<unknown> {
        return JSON.parse(this._bodyText);
    }
}

/** Upload data with progress tracking, returning a Promise.
 *
 * Minimal fetch-like wrapper around XMLHttpRequest.
 */
export function upload(options: UploadOptions): Promise<UploadResponse> {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const method = options.method ?? 'POST';
        const url = options.url.startsWith('http') ? options.url : API_BASE + options.url;

        xhr.open(method, url);

        if (options.headers) {
            for (const [key, value] of Object.entries(options.headers)) {
                xhr.setRequestHeader(key, value);
            }
        }

        if (options.onProgress && xhr.upload) {
            xhr.upload.addEventListener('progress', (event) => {
                options.onProgress!({
                    loaded: event.loaded,
                    total: event.total
                });
            });
        }

        // Chunk flushing logic — shared between progress and load events.
        // The load event fires when the response is complete, but the last
        // progress event may not have delivered the final NDJSON lines.
        let _parsedLength = 0;
        const _flushChunks = (finalize: boolean) => {
            const text = xhr.responseText;
            if (text.length <= _parsedLength) {
                return;
            }

            const newText = text.substring(_parsedLength);
            _parsedLength = text.length;

            // Split into lines.  During progress events, only yield complete
            // lines (ending with \n) — the last segment may be a partial line
            // that hasn't fully arrived yet.  When finalizing (load event),
            // process ALL lines including the last one, since the response is
            // fully received.
            const lines = newText.split('\n');
            const complete = finalize || newText.endsWith('\n') ? lines : lines.slice(0, -1);
            for (const line of complete) {
                if (line.length > 0) {
                    options.onChunk!(line);
                }
            }
        };

        if (options.onChunk) {
            xhr.addEventListener('progress', () => _flushChunks(false));
        }

        xhr.addEventListener('load', () => {
            // Flush any remaining chunks.  finalize=true ensures the last
            // line is processed even if it doesn't end with \n (e.g. a
            // non-streaming error response or the final NDJSON event).
            if (options.onChunk) {
                _flushChunks(true);
            }

            resolve(
                new UploadResponse(
                    xhr.status,
                    xhr.statusText,
                    xhr.responseURL || url,
                    xhr.responseText
                )
            );
        });

        xhr.addEventListener('error', () => {
            reject(new TypeError('Network request failed'));
        });

        xhr.addEventListener('abort', () => {
            reject(new DOMException('The user aborted a request.', 'AbortError'));
        });

        if (options.signal) {
            if (options.signal.aborted) {
                xhr.abort();
                return;
            }
            options.signal.addEventListener('abort', () => {
                xhr.abort();
            });
        }

        xhr.send(options.body ?? null);
    });
}
