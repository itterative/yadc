import { API_BASE, apiErrorMessage } from '$lib/api';

export interface ServerInfo {
    platform: string;
}

export async function fetchInfo(signal?: AbortSignal): Promise<ServerInfo> {
    const res = await fetch(`${API_BASE}/api/info`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}
