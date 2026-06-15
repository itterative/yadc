import { API_BASE, apiErrorMessage } from '$lib/api';
import { debounce } from '$lib/async';
import type { EnvInfo, EnvListResult } from './store';

async function _fetchEnvs(signal?: AbortSignal): Promise<EnvInfo[]> {
    const res = await fetch(`${API_BASE}/api/envs`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced env list fetch — dedupes simultaneous manual and SSE-driven refreshes. */
export const fetchEnvs = debounce(_fetchEnvs);

export async function saveEnv(
    name: string,
    /**
     * Fields to update on the env.
     * - omitted from the object: leave the existing value untouched
     * - set to a string/number: store the new value
     * - set to `null`: clear the field (mirrors `yadc envs delete <key>`)
     */
    data: {
        api_url?: string | null;
        api_token?: string | null;
        api_model_name?: string | null;
        max_concurrent?: number | null;
    },
    signal?: AbortSignal
): Promise<EnvInfo> {
    const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export async function deleteEnv(name: string, signal?: AbortSignal): Promise<void> {
    const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}`, {
        method: 'DELETE',
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

async function _fetchModels(name: string, signal?: AbortSignal): Promise<EnvListResult> {
    // GET-only. The ``yadc_password`` session cookie is auto-attached by
    // the browser, so a single GET carries everything needed (the
    // backend's ``resolve_request_password`` reads the cookie and falls
    // back to the ``YADC_PASSWORD`` env var).
    const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}/models`, { signal });
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
    signal?: AbortSignal
): Promise<{ value: string }> {
    // The ``yadc_password`` session cookie is auto-attached by the
    // browser; no body is needed.
    const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}/reveal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key }),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export async function fetchKeyMode(signal?: AbortSignal): Promise<{
    mode: 'keyring' | 'password';
    env_password_set: boolean;
}> {
    const res = await fetch(`${API_BASE}/api/envs/key-mode`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Switch the key storage mode, or change the password for password mode.
 *
 *  - `password` is the **new** password (sent in the body — it's a
 *    value being submitted, not a credential being presented). Pass
 *    `undefined` to switch to keyring mode.
 *  - The **current** password comes from the ``yadc_password`` session
 *    cookie (with the ``YADC_PASSWORD`` env-var fallback). It's
 *    needed when switching FROM password mode. */
export async function setKeyMode(
    mode: 'keyring' | 'password',
    password?: string,
    signal?: AbortSignal
): Promise<{ mode: 'keyring' | 'password' }> {
    const body: Record<string, unknown> = { mode };
    if (password !== undefined) {
        body.password = password;
    }
    const res = await fetch(`${API_BASE}/api/envs/key-mode`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Convenience: change the key storage password without switching mode. */
export async function changeKeyPassword(
    newPassword: string,
    signal?: AbortSignal
): Promise<{
    mode: 'keyring' | 'password';
}> {
    return setKeyMode('password', newPassword, signal);
}
