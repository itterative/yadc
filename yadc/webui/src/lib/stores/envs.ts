import { API_BASE, apiErrorMessage } from '$lib/api';
import { debounce } from '$lib/async';
import { get } from 'svelte/store';
import { writable, readonly, type Readable } from 'svelte/store';
import { sessionPassword } from './sessionPassword';

// --- Types matching the backend API ---

export interface EnvInfo {
    name: string;
    api_url: string | null;
    api_token: string | null; // masked as [REDACTED]
    api_model_name: string | null;
    has_token: boolean;
    token_method: 'none' | 'keyring' | 'password';
}

export interface EnvListResult {
    models: string[];
    default?: string;
}

// --- Reactive store ---

export interface EnvStoreState {
    loaded: boolean;
    items: EnvInfo[];
}

const _envs = writable<EnvStoreState>({ loaded: false, items: [] });

/** Reactive store for environments. */
export const envs: Readable<EnvStoreState> = readonly(_envs);

/** Fetch all environments from the API and update the store. */
export async function refreshEnvs(): Promise<EnvInfo[]> {
    const items = await fetchEnvs();
    _envs.set({ loaded: true, items });
    return items;
}

// --- API helpers ---

async function _fetchEnvs(): Promise<EnvInfo[]> {
    const res = await fetch(`${API_BASE}/api/envs`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced env list fetch — dedupes simultaneous manual and SSE-driven refreshes. */
export const fetchEnvs = debounce(_fetchEnvs);

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

async function _fetchModels(name: string): Promise<EnvListResult> {
    const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}/models`, {
        method: 'POST'
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced model list fetch — dedupes rapid env selection changes. */
export const fetchModels = debounce(_fetchModels);

export async function revealEnvValue(
    name: string,
    key: string,
    password?: string | null
): Promise<{ value: string }> {
    const body: Record<string, unknown> = { key };
    const resolvedPassword = password !== undefined ? password : get(sessionPassword);
    if (resolvedPassword) {
        body.password = resolvedPassword;
    }
    const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}/reveal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export async function fetchKeyMode(): Promise<{
    mode: 'keyring' | 'password';
    env_password_set: boolean;
}> {
    const res = await fetch(`${API_BASE}/api/envs/key-mode`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export async function setKeyMode(
    mode: 'keyring' | 'password',
    password?: string,
    oldPassword?: string
): Promise<{ mode: 'keyring' | 'password' }> {
    const body: Record<string, unknown> = { mode };
    if (password !== undefined) {
        body.password = password;
    }
    if (oldPassword !== undefined) {
        body.old_password = oldPassword;
    }
    const res = await fetch(`${API_BASE}/api/envs/key-mode`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Convenience: change password without switching mode. */
export async function changeKeyPassword(
    oldPassword: string,
    newPassword: string
): Promise<{ mode: 'keyring' | 'password' }> {
    return setKeyMode('password', newPassword, oldPassword);
}
