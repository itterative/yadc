import { writable, readonly, type Readable } from 'svelte/store';
import { fetchInfo } from './api';

export interface InfoStoreState {
    loaded: boolean;
    platform: string | null;
}

const _info = writable<InfoStoreState>({ loaded: false, platform: null });

/** Reactive store for server info. */
export const info: Readable<InfoStoreState> = readonly(_info);

/** Module-level cache for non-reactive access (e.g. ``isWindows()``). */
let _serverPlatform: string | null = null;

/** Fetch server info and update the store. */
export async function refreshInfo(signal?: AbortSignal): Promise<string | null> {
    let platform: string | null = null;
    try {
        const data = await fetchInfo(signal);
        platform = data.platform;
        _serverPlatform = platform;
    } finally {
        if (!signal?.aborted) {
            _info.set({ loaded: true, platform });
        }
    }
    return platform;
}

/** Return the cached server platform (``null`` before first fetch). */
export function getServerPlatform(): string | null {
    return _serverPlatform;
}
