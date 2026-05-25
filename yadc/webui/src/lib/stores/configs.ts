import { API_BASE, apiErrorMessage } from '$lib/api';

// --- Types matching the backend API ---

export interface ExportBackend {
	name: string;
	description: string;
	formats: string[];
}

export interface ExportResult {
	status: string;
	count: number;
	dataset: string;
	backend: string;
	format: string;
	source: string;
	output: string;
}

export interface DatasetConfig {
	name: string;
	config_path: string;
}

export interface DatasetConfigDetail {
	name: string;
	config_path: string;
	content: string;
	parsed: Record<string, unknown>;
}

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

export async function fetchConfig(name: string): Promise<DatasetConfigDetail> {
	const res = await fetch(`${API_BASE}/api/configs/${encodeURIComponent(name)}`);
	if (!res.ok) {
		throw new Error(await apiErrorMessage(res));
	}
	return res.json();
}

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
	patch: Record<string, unknown>
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
	patch: Record<string, unknown>
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

export async function deleteConfig(name: string): Promise<void> {
	const res = await fetch(`${API_BASE}/api/configs/${encodeURIComponent(name)}`, {
		method: 'DELETE'
	});
	if (!res.ok) {
		throw new Error(await apiErrorMessage(res));
	}
}
