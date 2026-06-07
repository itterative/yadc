import { writable, readonly, type Readable } from 'svelte/store';
import { fetchEnvs } from './api';

// --- Types matching the backend API ---

export interface EnvInfo {
    name: string;
    api_url: string | null;
    api_token: string | null; // masked as [REDACTED]
    api_model_name: string | null;
    /** Default concurrency for batch captioning. ``null`` = unset (sequential). */
    max_concurrent: number | null;
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
