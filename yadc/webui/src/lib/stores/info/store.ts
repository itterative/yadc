import { writable, readonly, derived, type Readable } from 'svelte/store';
import { fetchInfo } from './api';

export interface InfoStoreState {
    loaded: boolean;
    platform: string | null;
}

const _info = writable<InfoStoreState>({ loaded: false, platform: null });

/** Reactive store for server info. */
export const info: Readable<InfoStoreState> = readonly(_info);

/** Reactive ``true`` when the server platform is Windows. */
export const isWindows: Readable<boolean> = derived(_info, ($info) => $info.platform === 'win32');

/** Fetch server info and update the store. */
export async function refreshInfo(signal?: AbortSignal): Promise<string | null> {
    let platform: string | null = null;
    try {
        const data = await fetchInfo(signal);
        platform = data.platform;
    } finally {
        if (!signal?.aborted) {
            _info.set({ loaded: true, platform });
        }
    }
    return platform;
}
