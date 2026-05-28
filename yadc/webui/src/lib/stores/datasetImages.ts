import { API_BASE, apiErrorMessage, friendlyErrorMessage } from '$lib/api';
import { upload, type UploadProgress } from '$lib/upload';
import { debounce } from '$lib/async';
import { sessionPassword } from './sessionPassword';
import { writable } from 'svelte/store';

// --- Types matching the backend API dataclasses ---

export interface DatasetInfo {
    name: string;
    config_path: string | null;
    image_count: number;
    has_caption: number;
    has_toml: number;
    last_scanned_t: number | null;
    first_image_id: number | null;
}

export interface ImageInfo {
    id: number;
    file_name: string;
    path: string;
    has_caption: boolean;
    has_toml: boolean;
    width: number;
    height: number;
    draft_names: string[];
    last_modified_t: number | null;
    caption_error?: string;
    flash?: number;
}

export interface ImagePage {
    images: ImageInfo[];
    next_token: string | null;
}

export interface DatasetUploadResult {
    dataset: DatasetInfo;
    warnings: string[];
}

export interface CaptionData {
    caption: string;
    extras: Record<string, unknown>;
    extras_raw?: string;
    drafts: Record<string, string>;
}

export interface HistoryEntry {
    index: number;
    caption: string;
    extras: Record<string, unknown>;
}

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

/** Upload files to create a new dataset. */
export async function uploadDataset(
    name: string,
    files: File[],
    onProgress?: (progress: UploadProgress) => void,
    signal?: AbortSignal
): Promise<DatasetUploadResult> {
    const formData = new FormData();
    formData.append('name', name);
    for (const file of files) {
        const filename = file.webkitRelativePath || file.name;
        formData.append('files', file, filename);
    }

    const res = await upload({
        url: '/api/datasets/upload',
        body: formData,
        onProgress,
        signal
    });

    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json() as Promise<DatasetUploadResult>;
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
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/caption`,
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
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/history/${historyIndex}/restore`,
        { method: 'PUT', headers: { 'Content-Type': 'application/json' } }
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
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/extras`,
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

export interface PromptPreview {
    system_prompt: string;
    user_prompt: string;
    template_context: Record<string, unknown>;
    template_context_toml: string;
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

export interface CaptioningJobInfo {
    status: 'idle' | 'running' | 'stopping' | 'error' | 'done';
    dataset_name: string;
    job_id: string;
    processed: number;
    total: number;
    errors: number;
    error: string | null;
    error_messages: string[];
}

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

// --- Store for paginated image browsing ---

export interface DatasetBrowserState {
    images: ImageInfo[];
    isLoading: boolean;
    isLoadingMore: boolean;
    error: string | null;
    hasMore: boolean;
}

export function createDatasetBrowserStore(datasetName: string, pageSize = 50) {
    const { subscribe, set, update } = writable<DatasetBrowserState>({
        images: [],
        isLoading: false,
        isLoadingMore: false,
        error: null,
        hasMore: true
    });

    let lastAfterId = 0;

    async function loadInitial() {
        lastAfterId = 0;
        update((s) => ({ ...s, isLoading: true, error: null }));

        try {
            const page = await fetchImages(datasetName, { limit: pageSize, afterId: 0 });

            if (page.images.length > 0) {
                lastAfterId = page.images[page.images.length - 1].id;
            }

            set({
                images: page.images,
                isLoading: false,
                isLoadingMore: false,
                error: null,
                hasMore: page.next_token !== null
            });
        } catch (e) {
            update((s) => ({
                ...s,
                isLoading: false,
                error: friendlyErrorMessage(e, 'Failed to load images')
            }));
        }
    }

    async function loadMore() {
        let currentState: DatasetBrowserState | undefined;
        update((s) => {
            currentState = s;
            return { ...s, isLoadingMore: true };
        });

        if (!currentState || currentState.isLoadingMore || !currentState.hasMore) {
            return;
        }

        try {
            const page = await fetchImages(datasetName, { limit: pageSize, afterId: lastAfterId });

            if (page.images.length > 0) {
                lastAfterId = page.images[page.images.length - 1].id;
            }

            update((s) => ({
                ...s,
                images: [...s.images, ...page.images],
                isLoadingMore: false,
                hasMore: page.next_token !== null
            }));
        } catch (e) {
            update((s) => ({
                ...s,
                isLoadingMore: false,
                error: friendlyErrorMessage(e, 'Failed to load more images')
            }));
        }
    }

    function updateImage(imageId: number, patch: Partial<ImageInfo>) {
        update((s) => ({
            ...s,
            images: s.images.map((img) => (img.id === imageId ? { ...img, ...patch } : img))
        }));
    }

    return {
        subscribe,
        loadInitial,
        loadMore,
        updateImage
    };
}
