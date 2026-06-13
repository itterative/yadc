import { API_BASE, apiErrorMessage } from '$lib/api';
import { upload, type UploadProgress } from '$lib/upload';
import { debounce } from '$lib/async';
import { clientId } from '../events';
import { sessionPassword } from '../sessionPassword';
import type {
    CaptionData,
    CaptioningJobInfo,
    DatasetFolder,
    DatasetInfo,
    DatasetUploadResult,
    DraftSummary,
    HistoryEntry,
    ImagePage,
    PromptPreview,
    UploadProgressEvent
} from './types';

// --- API helpers ---

async function _fetchDatasets(signal?: AbortSignal): Promise<DatasetInfo[]> {
    const res = await fetch(`${API_BASE}/api/datasets`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced dataset list fetch — dedupes rapid navigation between pages. */
export const fetchDatasets = debounce(_fetchDatasets);

/** Import an existing TOML config as a new dataset. */
export async function importDataset(
    name: string,
    tomlPath: string,
    signal?: AbortSignal
): Promise<DatasetInfo> {
    const res = await fetch(`${API_BASE}/api/datasets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, toml_path: tomlPath }),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Create a new dataset from image directory paths. */
export async function createDataset(
    name: string,
    imagePaths: string[],
    signal?: AbortSignal
): Promise<DatasetInfo> {
    const res = await fetch(`${API_BASE}/api/datasets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, image_paths: imagePaths }),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Upload files to create a new dataset.
 *
 *  The server returns a streaming NDJSON response with progress events.
 *  Use `onEvent` to receive intermediate progress (validation, writing phases).
 *  The promise resolves with the final result or rejects on error.
 */
export async function uploadDataset(
    name: string,
    files: File[],
    onProgress?: (progress: UploadProgress) => void,
    onEvent?: (event: UploadProgressEvent) => void,
    signal?: AbortSignal
): Promise<DatasetUploadResult> {
    const formData = new FormData();
    formData.append('name', name);
    for (const file of files) {
        const filename = file.webkitRelativePath || file.name;
        formData.append('files', file, filename);
    }

    let settled = false;

    return new Promise<DatasetUploadResult>((resolve, reject) => {
        upload({
            url: `/api/datasets/upload?source=${encodeURIComponent(clientId)}`,
            body: formData,
            onProgress,
            signal,
            onChunk: (line) => {
                try {
                    const event = JSON.parse(line) as UploadProgressEvent;
                    if (event.phase === 'complete') {
                        settled = true;
                        resolve({ dataset: event.dataset!, warnings: event.warnings ?? [] });
                    } else if (event.phase === 'error') {
                        settled = true;
                        reject(new Error(event.message));
                    } else if (onEvent) {
                        onEvent(event);
                    }
                } catch {
                    // Ignore unparseable lines
                }
            }
        })
            .then(async (res) => {
                if (settled) {
                    return;
                }
                if (!res.ok) {
                    reject(new Error(await apiErrorMessage(res)));
                } else {
                    reject(new Error('Upload completed without a result'));
                }
            })
            .catch((e) => {
                if (!settled) {
                    reject(e);
                }
            });
    });
}

/** Upload files to append to an existing managed dataset.
 *
 *  Same streaming NDJSON contract as `uploadDataset`.
 */
export async function appendUploadDataset(
    name: string,
    files: File[],
    onProgress?: (progress: UploadProgress) => void,
    onEvent?: (event: UploadProgressEvent) => void,
    signal?: AbortSignal
): Promise<DatasetUploadResult> {
    const formData = new FormData();
    for (const file of files) {
        const filename = file.webkitRelativePath || file.name;
        formData.append('files', file, filename);
    }

    let settled = false;

    return new Promise<DatasetUploadResult>((resolve, reject) => {
        upload({
            url: `/api/datasets/${encodeURIComponent(name)}/upload?source=${encodeURIComponent(clientId)}`,
            body: formData,
            onProgress,
            signal,
            onChunk: (line) => {
                try {
                    const event = JSON.parse(line) as UploadProgressEvent;
                    if (event.phase === 'complete') {
                        settled = true;
                        resolve({ dataset: event.dataset!, warnings: event.warnings ?? [] });
                    } else if (event.phase === 'error') {
                        settled = true;
                        reject(new Error(event.message));
                    } else if (onEvent) {
                        onEvent(event);
                    }
                } catch {
                    // Ignore unparseable lines
                }
            }
        })
            .then(async (res) => {
                if (settled) {
                    return;
                }
                if (!res.ok) {
                    reject(new Error(await apiErrorMessage(res)));
                } else {
                    reject(new Error('Append completed without a result'));
                }
            })
            .catch((e) => {
                if (!settled) {
                    reject(e);
                }
            });
    });
}

/** Commit a staged upload after conflict resolution.
 *
 *  Streaming NDJSON response: committing → complete/error.
 */
export async function commitStagingUpload(
    name: string,
    stagingId: string,
    resolutions: Record<string, string>,
    onEvent?: (event: UploadProgressEvent) => void,
    signal?: AbortSignal
): Promise<DatasetUploadResult> {
    const body = JSON.stringify({ staging_id: stagingId, resolutions });

    let settled = false;

    return new Promise<DatasetUploadResult>((resolve, reject) => {
        upload({
            url: `/api/datasets/${encodeURIComponent(name)}/staging/commit?source=${encodeURIComponent(clientId)}`,
            body,
            headers: { 'Content-Type': 'application/json' },
            signal,
            onChunk: (line) => {
                try {
                    const event = JSON.parse(line) as UploadProgressEvent;
                    if (event.phase === 'complete') {
                        settled = true;
                        resolve({ dataset: event.dataset!, warnings: event.warnings ?? [] });
                    } else if (event.phase === 'error') {
                        settled = true;
                        reject(new Error(event.message));
                    } else if (onEvent) {
                        onEvent(event);
                    }
                } catch {
                    // Ignore unparseable lines
                }
            }
        })
            .then(async (res) => {
                if (settled) {
                    return;
                }
                if (!res.ok) {
                    reject(new Error(await apiErrorMessage(res)));
                } else {
                    reject(new Error('Commit completed without a result'));
                }
            })
            .catch((e) => {
                if (!settled) {
                    reject(e);
                }
            });
    });
}

/** Delete files and/or folders from a managed dataset.
 *
 *  Only works for datasets with `source === 'upload'`.
 */
export async function deleteDatasetItems(
    name: string,
    paths: string[],
    signal?: AbortSignal
): Promise<{ deleted: string[]; warnings: string[] }> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(name)}/items?source=${encodeURIComponent(clientId)}`,
        {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paths }),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** List folders for a managed dataset with image counts. */
export async function fetchFolders(name: string, signal?: AbortSignal): Promise<DatasetFolder[]> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(name)}/folders`, {
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** List draft names with image counts for a dataset. */
export async function fetchDraftSummary(
    name: string,
    signal?: AbortSignal
): Promise<DraftSummary[]> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(name)}/drafts/summary`, {
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Delete a named draft from all images in a dataset. */
export async function deleteDraftAll(
    name: string,
    draftName: string,
    signal?: AbortSignal
): Promise<number> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(name)}/drafts/${encodeURIComponent(draftName)}?source=${encodeURIComponent(clientId)}`,
        { method: 'DELETE', signal }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    const data = await res.json();
    return data.deleted;
}

/** Delete/unregister a dataset. */
export async function deleteDataset(name: string, signal?: AbortSignal): Promise<void> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(name)}`, {
        method: 'DELETE',
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

/** Rescan a dataset's images from disk. Returns updated dataset info. */
export async function rescanDataset(name: string, signal?: AbortSignal): Promise<DatasetInfo> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(name)}/rescan?source=${encodeURIComponent(clientId)}`,
        { method: 'POST', signal }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Duplicate a managed dataset to a new name.
 *
 *  Defaults to `mode='hardlink'` (image bytes are hardlinked,
 *  config + sidecars are copied). The service probes the actual
 *  destination dir; if hardlinks aren't supported, the endpoint
 *  returns 409 after cleaning up the partial new dir. The
 *  frontend should catch this case and offer to retry with
 *  `mode='copy'`.
 */
export async function duplicateDataset(
    name: string,
    newName: string,
    mode: 'copy' | 'hardlink' = 'hardlink',
    signal?: AbortSignal
): Promise<DatasetInfo> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(name)}/duplicate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_name: newName, mode }),
        signal
    });
    if (!res.ok) {
        // Attach the HTTP status so callers can branch on specific
        // status codes (e.g. 409 hardlink-not-supported).
        const message = await apiErrorMessage(res);
        const err = new Error(message);
        (err as Error & { status?: number }).status = res.status;
        throw err;
    }
    return res.json();
}

async function _fetchImages(
    datasetName: string,
    options: { limit?: number; next?: string } = {},
    signal?: AbortSignal
): Promise<ImagePage> {
    const params = new URLSearchParams();
    if (options.limit) {
        params.set('limit', String(options.limit));
    }
    if (options.next) {
        // ``next`` is an opaque cursor — pass it through as-is.
        params.set('next', options.next);
    }

    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images?${params}`,
        { signal }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced image fetch — dedupes rapid calls (e.g., tab switching). */
export const fetchImages = debounce(_fetchImages);

async function _fetchCaption(
    datasetName: string,
    imageId: number,
    signal?: AbortSignal
): Promise<CaptionData> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/caption`,
        { signal }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced caption fetch — dedupes rapid image selection changes. */
export const fetchCaption = debounce(_fetchCaption);

export async function updateCaption(
    datasetName: string,
    imageId: number,
    caption: string,
    signal?: AbortSignal
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/caption?source=${encodeURIComponent(clientId)}`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ caption }),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

async function _fetchHistory(
    datasetName: string,
    imageId: number,
    signal?: AbortSignal
): Promise<HistoryEntry[]> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/history`,
        { signal }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced history fetch — dedupes rapid image selection changes. */
export const fetchHistory = debounce(_fetchHistory);

export async function restoreHistory(
    datasetName: string,
    imageId: number,
    historyIndex: number,
    signal?: AbortSignal
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/history/${historyIndex}/restore?source=${encodeURIComponent(clientId)}`,
        { method: 'PUT', headers: { 'Content-Type': 'application/json' }, signal }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

export async function deleteHistory(
    datasetName: string,
    imageId: number,
    entryHash: string,
    signal?: AbortSignal
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/history/${entryHash}?source=${encodeURIComponent(clientId)}`,
        { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, signal }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

export async function deleteDraft(
    datasetName: string,
    imageId: number,
    draftName: string,
    signal?: AbortSignal
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/drafts/${encodeURIComponent(draftName)}?source=${encodeURIComponent(clientId)}`,
        { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, signal }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

export async function writeDraft(
    datasetName: string,
    imageId: number,
    draftName: string,
    content: string,
    signal?: AbortSignal
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/drafts/${encodeURIComponent(draftName)}?source=${encodeURIComponent(clientId)}`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content }),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

export async function updateExtras(
    datasetName: string,
    imageId: number,
    extrasRaw: string,
    signal?: AbortSignal
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/extras?source=${encodeURIComponent(clientId)}`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ extras_raw: extrasRaw }),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

export function thumbnailUrl(datasetName: string, imageId: number, size = 256): string {
    return `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/thumbnail?size=${size}`;
}

export async function fetchPromptPreview(
    datasetName: string,
    imageId: number,
    options: { template?: string; template_name?: string } = {},
    signal?: AbortSignal
): Promise<PromptPreview> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/preview-prompt`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(options),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export function mediaUrl(datasetName: string, imageId: number): string {
    return `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/media`;
}

// --- Captioning API helpers ---

/** Start a captioning job. Returns initial job info. */
export async function startCaptioning(
    datasetName: string,
    options: Record<string, unknown>,
    signal?: AbortSignal
): Promise<CaptioningJobInfo> {
    const password = sessionPassword.get();
    const body = password ? { ...options, password } : options;
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Fetch the current captioning status for a dataset. */
async function _fetchCaptioningStatus(
    datasetName: string,
    signal?: AbortSignal
): Promise<CaptioningJobInfo> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`, {
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced captioning status fetch — collapses rapid calls and dedupes in-flight requests. */
export const fetchCaptioningStatus = debounce(_fetchCaptioningStatus);

/** Stop a running captioning job. Raw API call — for toast-enabled version use `captionActions.stopCaptioning`. */
export async function stopCaptioning(datasetName: string, signal?: AbortSignal): Promise<boolean> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`, {
        method: 'DELETE',
        signal
    });
    return res.ok;
}

/** Start a single-image captioning job. Returns initial job info. */
export async function captionSingleImage(
    datasetName: string,
    imageId: number,
    options: Record<string, unknown> = {},
    signal?: AbortSignal
): Promise<CaptioningJobInfo> {
    const password = sessionPassword.get();
    const body = password ? { ...options, password } : options;
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/caption`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Refine a caption by sending feedback to the model.
 *  Sends the current caption (or provided one) plus user feedback as
 *  extra_messages to the model. Returns initial job info. */
export async function refineCaption(
    datasetName: string,
    imageId: number,
    feedback: string,
    caption: string,
    options: Record<string, unknown> = {},
    signal?: AbortSignal
): Promise<CaptioningJobInfo> {
    const password = sessionPassword.get();
    const body = {
        ...options,
        feedback,
        refine_caption: caption,
        ...(password ? { password } : {})
    };
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/refine`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Fetch the latest dry-run refine result for an image (if any). */
export async function fetchRefineResult(
    datasetName: string,
    imageId: number,
    source: 'caption' | 'draft' = 'caption',
    draftName: string = '',
    signal?: AbortSignal
): Promise<string | null> {
    const params = new URLSearchParams();
    if (source === 'draft' && draftName) {
        params.set('source', source);
        params.set('draft_name', draftName);
    }
    const qs = params.toString();
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/refine${qs ? '?' + qs : ''}`,
        { signal }
    );
    if (res.status === 404) {
        return null;
    }
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    const data = await res.json();
    return data.caption ?? null;
}

/** Evict the cached refine result for an image after the user accepts it.
 *
 * Called after a successful caption/draft write so the next open of the
 * refine dialog starts from a clean state. Treats 404 (nothing cached) and
 * 409 (a different value is cached — a newer refine) as non-errors and
 * returns ``false``. Throws on other failures.
 */
export async function deleteRefineResult(
    datasetName: string,
    imageId: number,
    caption: string,
    source: 'caption' | 'draft' = 'caption',
    draftName: string = '',
    signal?: AbortSignal
): Promise<boolean> {
    const password = sessionPassword.get();
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/refine`,
        {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                caption,
                source,
                draft_name: draftName,
                ...(password ? { password } : {})
            }),
            signal
        }
    );
    if (res.status === 404 || res.status === 409) {
        return false;
    }
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return true;
}
