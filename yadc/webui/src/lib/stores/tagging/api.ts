import { API_BASE, apiErrorMessage } from '$lib/api';
import { clientId } from '../events';
import type {
    ActiveTaggerResponse,
    CancelResult,
    SwapTaggerBody,
    SwapTaggerResponse,
    TagJobInfo,
    TaggerModelSummary,
    TaggerResult,
    TagSaveOptions
} from './types';

// --- Tagging API helpers ---
//
// The tagger endpoints do not require a password by default, so these are
// plain fetches (no ``withPasswordRetry``). Errors surface as toasts at the
// action layer — including a 503 "tagger not configured", whose message is
// readable enough to show directly.

/** Tag a single image synchronously. Returns the thresholded result
 *  immediately (used by the interactive Tag button — no job/SSE round-trip).
 *  503 when the tagger isn't configured. */
export async function tagImage(
    datasetName: string,
    imageId: number,
    options: {
        rating_threshold?: number;
        general_threshold?: number;
        character_threshold?: number;
        replace_underscores?: boolean;
        source?: string;
    } = {},
    signal?: AbortSignal
): Promise<TaggerResult> {
    const params = new URLSearchParams();
    if (options.source) {
        params.set('source', options.source);
    }
    const qs = params.toString();
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/tag${qs ? '?' + qs : ''}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rating_threshold: options.rating_threshold,
                general_threshold: options.general_threshold,
                character_threshold: options.character_threshold,
                replace_underscores: options.replace_underscores
            }),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Start a batch tagging job. Returns initial job info (``202``).
 *  ``imageIds`` omitted/empty → whole dataset; otherwise just those.
 *  Thresholds fall back to the server config when undefined. */
export async function startTagJob(
    datasetName: string,
    options: {
        image_ids?: number[];
        rating_threshold?: number;
        general_threshold?: number;
        character_threshold?: number;
        replace_underscores?: boolean;
        save?: TagSaveOptions;
        source?: string;
    } = {},
    signal?: AbortSignal
): Promise<TagJobInfo> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/tag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(options),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Stop a running tagging job. Raw API call — for toast-enabled version
 *  use ``taggingActions.stopTagging``. */
export async function stopTagJob(datasetName: string, signal?: AbortSignal): Promise<boolean> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/tag`, {
        method: 'DELETE',
        signal
    });
    return res.ok;
}

/** Cancel a tagging job and/or interrupt an in-flight tag (graceful-first,
 *  kill-as-fallback). Pass a ``jobId`` to cancel a specific batch job;
 *  omit it for the single-image case (cancel escalates straight to killing
 *  the subprocess if a request is mid-inference). Returns the outcome. */
export async function cancelTagging(jobId?: string, signal?: AbortSignal): Promise<CancelResult> {
    const res = await fetch(`${API_BASE}/api/tagging/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId ?? null }),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Fetch the current tagging job status for a dataset. */
export async function fetchTagJobStatus(
    datasetName: string,
    signal?: AbortSignal
): Promise<TagJobInfo> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/tag`, {
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Write user-pruned tags for an image to a draft / extras without
 *  re-running the model. Used by the interactive Tags tab's save bar.
 *  The body echoes the ``TaggerResult`` shape so the backend persists
 *  exactly what the user kept after pruning. */
export async function saveImageTags(
    datasetName: string,
    imageId: number,
    result: TaggerResult,
    save: TagSaveOptions,
    signal?: AbortSignal
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/tags?source=${encodeURIComponent(clientId)}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tags: result.tags, categories: result.categories, save }),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

/** Read the cached tag result for an image from the backend LRU.
 *  Returns ``null`` when no entry exists (never tagged, evicted by LRU,
 *  or tagged under settings that no longer match the current config).
 *  Used by the interactive Tags tab on a fresh page load to surface
 *  results produced by an earlier batch job without re-running the
 *  model. The thresholds + replace_underscores options are passed
 *  through to the backend so the GET lands in the same cache bucket
 *  as the original POST /tag write. */
export async function fetchTagResult(
    datasetName: string,
    imageId: number,
    options: {
        rating_threshold?: number | null;
        general_threshold?: number | null;
        character_threshold?: number | null;
        replace_underscores?: boolean | null;
    } = {},
    signal?: AbortSignal
): Promise<TaggerResult | null> {
    const params = new URLSearchParams();
    if (options.rating_threshold != null) {
        params.set('rating_threshold', String(options.rating_threshold));
    }
    if (options.general_threshold != null) {
        params.set('general_threshold', String(options.general_threshold));
    }
    if (options.character_threshold != null) {
        params.set('character_threshold', String(options.character_threshold));
    }
    if (options.replace_underscores != null) {
        params.set('replace_underscores', options.replace_underscores ? 'true' : 'false');
    }
    const qs = params.toString();
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/tag${qs ? '?' + qs : ''}`,
        { signal }
    );
    if (res.status === 404) {
        return null;
    }
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as TaggerResult;
}

/** Preview what :func:`saveImageTags` would write, without touching disk.
 *  Returns formatter text (draft) / the ``[tags]`` TOML sub-table (extras) /
 *  empty string (none). */
export async function previewImageTags(
    datasetName: string,
    imageId: number,
    result: TaggerResult,
    save: TagSaveOptions,
    signal?: AbortSignal
): Promise<string> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/tags/preview`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tags: result.tags, categories: result.categories, save }),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    const data = (await res.json()) as { content: string };
    return data.content;
}

/** Context-free format preview — no dataset or image required. Used by
 *  settings panels to show what a save produces from fixed mock data.
 *  Same return shape as :func:`previewImageTags`. */
export async function previewTagFormats(
    result: TaggerResult,
    save: TagSaveOptions,
    signal?: AbortSignal
): Promise<string> {
    const res = await fetch(`${API_BASE}/api/tagging/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags: result.tags, categories: result.categories, save }),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    const data = (await res.json()) as { content: string };
    return data.content;
}

// --- Tagger model swap ---

/** Fetch the persisted active tagger selection + subprocess liveness.
 *  Used by the SettingsDialog picker to show "Currently running: …". */
export async function fetchActiveTagger(signal?: AbortSignal): Promise<ActiveTaggerResponse> {
    const res = await fetch(`${API_BASE}/api/tagger/active`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as ActiveTaggerResponse;
}

export interface TaggerModelsResponse {
    models: TaggerModelSummary[];
    local: TaggerModelSummary;
    /** Names of supported preproc profiles (mirrors
     *  ``yadc.taggers.onnx_preprocess.list_profiles``). Drives the
     *  Profile dropdown in the picker. */
    profiles: string[];
}

/** Fetch the curated model catalog (SmilingWolf HF repos + the local-file sentinel).
 *  The picker dropdown renders from this; the user picks one and we POST to
 *  ``swapTaggerModel`` with a fully-formed selection. */
export async function listTaggerModels(signal?: AbortSignal): Promise<TaggerModelsResponse> {
    const res = await fetch(`${API_BASE}/api/tagger/models`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as TaggerModelsResponse;
}

/** Discriminated result from :func:`swapTaggerModel`. The action layer
 *  matches on ``status`` to pick the right toast (and never has to
 *  re-parse the response body or distinguish 409 from 429). */
export type SwapTaggerResult =
    | { status: 'ok'; response: SwapTaggerResponse }
    | { status: 'busy'; message: string }
    | { status: 'in_progress'; message: string; retryAfterS: number }
    | { status: 'error'; message: string };

/** Swap the active tagger selection. Maps the backend's status codes into a
 *  small set of variants the UI can switch on without re-parsing the body:
 *  200 → ``ok``; 409 → ``busy`` (batch running); 429 → ``in_progress``
 *  (concurrent swap, with retry hint); anything else → ``error``. */
export async function swapTaggerModel(
    body: SwapTaggerBody,
    signal?: AbortSignal
): Promise<SwapTaggerResult> {
    const res = await fetch(`${API_BASE}/api/tagger/swap`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal
    });
    if (res.ok) {
        const response = (await res.json()) as SwapTaggerResponse;
        return { status: 'ok', response };
    }
    let parsed: { error?: string; retry_after_s?: number } | null = null;
    try {
        parsed = (await res.json()) as { error?: string; retry_after_s?: number };
    } catch {
        /* non-JSON body */
    }
    const message = parsed?.error || `Swap failed (HTTP ${res.status})`;
    if (res.status === 409) {
        return { status: 'busy', message };
    }
    if (res.status === 429) {
        return { status: 'in_progress', message, retryAfterS: parsed?.retry_after_s ?? 2 };
    }
    return { status: 'error', message };
}
