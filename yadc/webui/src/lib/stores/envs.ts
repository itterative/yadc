import { API_BASE, apiErrorMessage } from '$lib/api';
import { writable, readonly, type Readable } from 'svelte/store';

// --- Types matching the backend API ---

export interface EnvInfo {
	name: string;
	api_url: string | null;
	api_token: string | null; // masked as [REDACTED]
	api_model_name: string | null;
}

export interface EnvListResult {
	models: string[];
	default?: string;
}

// --- Reactive store ---

export interface EnvStoreState {
	loaded: boolean;
	items: string[];
}

const _envs = writable<EnvStoreState>({ loaded: false, items: [] });

/** Reactive store for environment names. */
export const envs: Readable<EnvStoreState> = readonly(_envs);

/** Fetch the environment list from the API and update the store. */
export async function refreshEnvs(): Promise<string[]> {
	const names = await fetchEnvs();
	_envs.set({ loaded: true, items: names });
	return names;
}

// --- API helpers ---

export async function fetchEnvs(): Promise<string[]> {
	const res = await fetch(`${API_BASE}/api/envs`);
	if (!res.ok) {
		throw new Error(await apiErrorMessage(res));
	}
	return res.json();
}

export async function fetchEnv(name: string): Promise<EnvInfo> {
	const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}`);
	if (!res.ok) {
		throw new Error(await apiErrorMessage(res));
	}
	return res.json();
}

export async function saveEnv(
	name: string,
	data: { api_url?: string; api_token?: string; api_model_name?: string }
): Promise<EnvInfo> {
	const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(data)
	});
	if (!res.ok) {
		throw new Error(await apiErrorMessage(res));
	}
	return res.json();
}

export async function deleteEnv(name: string): Promise<void> {
	const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}`, {
		method: 'DELETE'
	});
	if (!res.ok) {
		throw new Error(await apiErrorMessage(res));
	}
}

export async function fetchModels(name: string): Promise<EnvListResult> {
	const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}/models`, {
		method: 'POST'
	});
	if (!res.ok) {
		throw new Error(await apiErrorMessage(res));
	}
	return res.json();
}
