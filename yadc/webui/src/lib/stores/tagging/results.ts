import { fetchTagResult } from './api';
import type { TaggerResult } from './types';

/** Handler invoked when the backend reports a tag result for an image.
 *  Two paths dispatch: the synchronous POST ``/tag`` response (via the
 *  caller's awaited value) and the SSE ``image_tagged`` event for
 *  batch-tagged images. The Tags tab registers a handler so a batch
 *  tagging the focused image updates the prune grid without a manual
 *  re-fetch.
 *
 *  The backend is the source of truth for persistence (see
 *  ``TaggingService._tag_results`` on the server) — these handlers are
 *  just a fan-out for active Tags tabs.
 */
export type TaggedHandler = (datasetName: string, imageId: number, result: TaggerResult) => void;

const _handlers: Set<TaggedHandler> = new Set();

/** Register a handler. Returns an unsubscribe function — call it from
 *  ``onMount`` cleanup so a destroyed component stops receiving
 *  events.
 */
export function onImageTagged(handler: TaggedHandler): () => void {
    _handlers.add(handler);
    return () => {
        _handlers.delete(handler);
    };
}

/** Fan out a tagged result to every registered handler. */
export function dispatchImageTagged(
    datasetName: string,
    imageId: number,
    result: TaggerResult
): void {
    for (const handler of _handlers) {
        handler(datasetName, imageId, result);
    }
}

/** Fetch the cached tag result for an image from the backend.
 *  Returns ``null`` when no entry is cached (never tagged, evicted
 *  by LRU, or tagged under different settings — see the backend's
 *  ``TaggerResultKey`` fingerprint). The caller passes the same
 *  threshold + replace_underscores options it would send to POST
 *  ``/tag`` so the read lands in the same cache bucket as the
 *  original write. The always-add / banned policy is resolved
 *  server-side per dataset — callers that need a policy change to
 *  surface on the displayed result refetch (the cache slot itself
 *  is unchanged; only the read-time transform differs). Used by the
 *  Tags tab on mount and image switch.
 */
export async function fetchCachedTagResult(
    datasetName: string,
    imageId: number,
    options: {
        rating_threshold?: number | null;
        general_threshold?: number | null;
        character_threshold?: number | null;
        replace_underscores?: boolean | null;
        per_tag_thresholds?: boolean | null;
        per_tag_column?: string | null;
    } = {},
    signal?: AbortSignal
): Promise<TaggerResult | null> {
    return fetchTagResult(datasetName, imageId, options, signal);
}
