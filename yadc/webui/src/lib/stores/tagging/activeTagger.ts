import { get, writable } from 'svelte/store';
import { fetchActiveTagger } from './api';
import { taggerStatus } from './taggerStatus';
import type { ActiveTaggerResponse } from './types';

/** Persisted active tagger selection + subprocess liveness + per-tag
 *  capability. Mirrors ``GET /api/tagger/active`` so components
 *  subscribe instead of fetching ad hoc. Fed by
 *  :func:`ensureActiveTaggerLoaded` on mount and re-fetched whenever
 *  the ``tagger_status`` SSE stream settles (swap done / failed) —
 *  cross-tab, unlike a window event. */
export const activeTagger = writable<ActiveTaggerResponse | null>(null);

/** Module-level single-flight promise so parallel mounters share one
 *  in-flight fetch. */
let _loadPromise: Promise<ActiveTaggerResponse> | null = null;
let _loaded = false;

/** Populate ``activeTagger`` from the server. Idempotent — the second
 *  call after the first resolves is a no-op. Throws on a non-200 so
 *  the caller can toast; the store keeps its current value on error. */
export async function ensureActiveTaggerLoaded(): Promise<ActiveTaggerResponse> {
    if (_loaded) {
        const current = get(activeTagger);
        if (current !== null) {
            return current;
        }
    }
    if (_loadPromise === null) {
        _loadPromise = (async () => {
            try {
                const response = await fetchActiveTagger();
                activeTagger.set(response);
                _loaded = true;
                return response;
            } catch (e) {
                _loadPromise = null;
                throw e;
            }
        })();
    }
    return _loadPromise;
}

/** Force-refresh from the server. Used when the subprocess settles
 *  after a swap (or in tests). */
export async function refreshActiveTagger(): Promise<ActiveTaggerResponse> {
    _loadPromise = null;
    _loaded = false;
    return ensureActiveTaggerLoaded();
}

// Re-fetch when the subprocess settles — a swap always ends in
// ``ready`` (new model live) or ``failed`` (rolled back), so the
// capability flags can't go stale after a minutes-long first-run HF
// download. Failures stay silent here (a stale value beats an
// unhandled rejection); the next mount retries.
taggerStatus.subscribe((status) => {
    if (status.state === 'ready' || status.state === 'failed') {
        void refreshActiveTagger().catch(() => {});
    }
});
