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
    onProgress?: (progress: UploadProgress) => void;
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

        xhr.addEventListener('load', () => {
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
