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

async function _fetchDatasets(): Promise<DatasetInfo[]> {
    const res = await fetch(`${API_BASE}/api/datasets`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced dataset list fetch — dedupes rapid navigation between pages. */
export const fetchDatasets = debounce(_fetchDatasets);

/** Import an existing TOML config as a new dataset. */
export async function importDataset(name: string, tomlPath: string): Promise<DatasetInfo> {
    const res = await fetch(`${API_BASE}/api/datasets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, toml_path: tomlPath })
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Create a new dataset from image directory paths. */
export async function createDataset(name: string, imagePaths: string[]): Promise<DatasetInfo> {
    const res = await fetch(`${API_BASE}/api/datasets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, image_paths: imagePaths })
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
    paths: string[]
): Promise<{ deleted: string[]; warnings: string[] }> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(name)}/items?source=${encodeURIComponent(clientId)}`,
        {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paths })
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** List folders for a managed dataset with image counts. */
export async function fetchFolders(name: string): Promise<DatasetFolder[]> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(name)}/folders`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** List draft names with image counts for a dataset. */
export async function fetchDraftSummary(name: string): Promise<DraftSummary[]> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(name)}/drafts/summary`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Delete a named draft from all images in a dataset. */
export async function deleteDraftAll(name: string, draftName: string): Promise<number> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(name)}/drafts/${encodeURIComponent(draftName)}?source=${encodeURIComponent(clientId)}`,
        { method: 'DELETE' }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    const data = await res.json();
    return data.deleted;
}

/** Delete/unregister a dataset. */
export async function deleteDataset(name: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(name)}`, {
        method: 'DELETE'
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

/** Rescan a dataset's images from disk. Returns updated dataset info. */
export async function rescanDataset(name: string): Promise<DatasetInfo> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(name)}/rescan?source=${encodeURIComponent(clientId)}`,
        { method: 'POST' }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

async function _fetchImages(
    datasetName: string,
    options: { limit?: number; afterId?: number } = {}
): Promise<ImagePage> {
    const params = new URLSearchParams();
    if (options.limit) {
        params.set('limit', String(options.limit));
    }
    if (options.afterId !== undefined) {
        params.set('after_id', String(options.afterId));
    }

    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images?${params}`
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced image fetch — dedupes rapid calls (e.g., tab switching). */
export const fetchImages = debounce(_fetchImages);

async function _fetchCaption(datasetName: string, imageId: number): Promise<CaptionData> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/caption`
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
    caption: string
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/caption?source=${encodeURIComponent(clientId)}`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ caption })
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

async function _fetchHistory(datasetName: string, imageId: number): Promise<HistoryEntry[]> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/history`
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
    historyIndex: number
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/history/${historyIndex}/restore?source=${encodeURIComponent(clientId)}`,
        { method: 'PUT', headers: { 'Content-Type': 'application/json' } }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

export async function deleteHistory(
    datasetName: string,
    imageId: number,
    entryHash: string
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/history/${entryHash}?source=${encodeURIComponent(clientId)}`,
        { method: 'DELETE', headers: { 'Content-Type': 'application/json' } }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

export async function deleteDraft(
    datasetName: string,
    imageId: number,
    draftName: string
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/drafts/${encodeURIComponent(draftName)}?source=${encodeURIComponent(clientId)}`,
        { method: 'DELETE', headers: { 'Content-Type': 'application/json' } }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

export async function updateExtras(
    datasetName: string,
    imageId: number,
    extrasRaw: string
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/extras?source=${encodeURIComponent(clientId)}`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ extras_raw: extrasRaw })
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
    options: { template?: string; template_name?: string } = {}
): Promise<PromptPreview> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/preview-prompt`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(options)
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
    options: Record<string, unknown>
): Promise<CaptioningJobInfo> {
    const password = sessionPassword.get();
    const body = password ? { ...options, password } : options;
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Fetch the current captioning status for a dataset. */
async function _fetchCaptioningStatus(datasetName: string): Promise<CaptioningJobInfo> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced captioning status fetch — collapses rapid calls and dedupes in-flight requests. */
export const fetchCaptioningStatus = debounce(_fetchCaptioningStatus);

/** Stop a running captioning job. Raw API call — for toast-enabled version use `captionActions.stopCaptioning`. */
export async function stopCaptioning(datasetName: string): Promise<boolean> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`, {
        method: 'DELETE'
    });
    return res.ok;
}

/** Start a single-image captioning job. Returns initial job info. */
export async function captionSingleImage(
    datasetName: string,
    imageId: number,
    options: Record<string, unknown> = {}
): Promise<CaptioningJobInfo> {
    const password = sessionPassword.get();
    const body = password ? { ...options, password } : options;
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/caption`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}
