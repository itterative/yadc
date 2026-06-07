import { API_BASE, apiErrorMessage } from '$lib/api';
import { debounce } from '$lib/async';
import type {
    Config,
    ConfigHistoryPage,
    DatasetConfig,
    DatasetConfigDetail,
    ExportBackend,
    ExportResult
} from './types';

// --- Export API helpers ---

export async function fetchExportBackends(): Promise<ExportBackend[]> {
    const res = await fetch(`${API_BASE}/api/export/backends`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export async function runExport(options: {
    dataset: string;
    backend?: string;
    format?: string;
    source?: string;
    draft?: string;
    with_drafts?: string[];
    output?: string;
    append?: boolean;
    caption_extension?: string;
}): Promise<ExportResult> {
    const res = await fetch(`${API_BASE}/api/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(options)
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

// --- Dataset drafts API ---

export async function fetchDatasetDrafts(datasetName: string): Promise<string[]> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/drafts`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

// --- Config API helpers ---

export async function fetchConfigs(): Promise<DatasetConfig[]> {
    const res = await fetch(`${API_BASE}/api/configs`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

async function _fetchConfig(name: string): Promise<DatasetConfigDetail> {
    const res = await fetch(`${API_BASE}/api/configs/${encodeURIComponent(name)}`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export const fetchConfig = debounce(_fetchConfig);

async function _fetchConfigHistory(
    name: string,
    options?: { limit?: number; next?: string | null }
): Promise<ConfigHistoryPage> {
    const params = new URLSearchParams();
    if (options?.limit) {
        params.set('limit', String(options.limit));
    }
    if (options?.next) {
        // ``next`` is an opaque cursor — pass it through as-is.
        // Falsy values (``''``, ``null``, ``undefined``) mean
        // "first page" and are omitted.
        params.set('next', options.next);
    }
    const qs = params.toString();
    const res = await fetch(
        `${API_BASE}/api/configs/${encodeURIComponent(name)}/history${qs ? '?' + qs : ''}`
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export const fetchConfigHistory = debounce(_fetchConfigHistory);

export async function updateConfig(name: string, content: string): Promise<DatasetConfigDetail> {
    const res = await fetch(`${API_BASE}/api/configs/${encodeURIComponent(name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content })
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export async function patchConfig(
    name: string,
    patch: Partial<Config>
): Promise<DatasetConfigDetail> {
    const res = await fetch(`${API_BASE}/api/configs/${encodeURIComponent(name)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch)
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export async function previewConfig(
    name: string,
    patch: Partial<Config>
): Promise<DatasetConfigDetail> {
    const res = await fetch(`${API_BASE}/api/configs/${encodeURIComponent(name)}?dry_run=true`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch)
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export async function restoreConfigHistory(
    name: string,
    entryId: number
): Promise<DatasetConfigDetail> {
    const res = await fetch(
        `${API_BASE}/api/configs/${encodeURIComponent(name)}/history/${entryId}/restore`,
        { method: 'POST' }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export async function deleteConfig(name: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/configs/${encodeURIComponent(name)}`, {
        method: 'DELETE'
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}
