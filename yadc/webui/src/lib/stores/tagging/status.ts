import { browser } from '$app/environment';
import { readonly, writable, type Readable, type Writable } from 'svelte/store';
import type { TagJobInfo } from './types';

/** A blank idle status — the shape a per-dataset slot starts from. */
export const INITIAL_TAGGING_STATUS: TagJobInfo = {
    status: 'idle',
    dataset_name: '',
    job_id: '',
    processed: 0,
    total: 0,
    errors: 0,
    error: null,
    error_messages: [],
    source: '',
    elapsed: 0
};

/** Statuses that mean the job is no longer making progress. An entry reaching
 *  one of these is evicted from the map after a grace period so the "Done" /
 *  "Error" badge flashes and then clears. */
const TERMINAL_STATUSES = new Set<TagJobInfo['status']>(['idle', 'done', 'error', 'cancelled']);

/** How long a terminal entry lingers before eviction. Mirrors the captioning
 *  status store — keeps per-dataset results visible briefly after a batch
 *  finishes. */
const TERMINAL_EVICTION_MS = 5 * 60 * 1000;

const _taggingStatuses: Writable<ReadonlyMap<string, TagJobInfo>> = writable(
    new Map<string, TagJobInfo>()
);

/** Per-dataset tagging job status, keyed by dataset name. Fed by the SSE
 *  ``tag_job_status`` event and the action functions' optimistic seeds. */
export const taggingStatuses: Readable<ReadonlyMap<string, TagJobInfo>> =
    readonly(_taggingStatuses);

const _evictionTimers = new Map<string, ReturnType<typeof setTimeout>>();

function _clearEviction(datasetName: string): void {
    const timer = _evictionTimers.get(datasetName);
    if (timer !== undefined) {
        clearTimeout(timer);
        _evictionTimers.delete(datasetName);
    }
}

function _scheduleEviction(datasetName: string): void {
    _clearEviction(datasetName);
    if (!browser) {
        return;
    }
    _evictionTimers.set(
        datasetName,
        setTimeout(() => {
            _evictionTimers.delete(datasetName);
            _taggingStatuses.update((map) => {
                if (!map.has(datasetName)) {
                    return map;
                }
                const next = new Map(map);
                next.delete(datasetName);
                return next;
            });
        }, TERMINAL_EVICTION_MS)
    );
}

/** Upsert a dataset's status. Terminal statuses are scheduled for eviction. */
export function setTaggingStatus(status: TagJobInfo): void {
    _taggingStatuses.update((map) => {
        const current = map.get(status.dataset_name);
        // Don't overwrite a terminal status from the same job with a
        // non-terminal one. The HTTP response from ``POST /tagger/start``
        // carries a snapshot read at submit time — for an all-cache-hit
        // batch the background task can finish (and dispatch the terminal
        // ``tag_job_status`` SSE event) before the response lands at the
        // client. A late response that says 'running' would otherwise
        // yank the UI back from 'done' to 'running'. The terminal SSE
        // event is the authoritative state for that job_id; if it never
        // arrives, the eviction timer cleans up after 5 min.
        if (
            current !== undefined &&
            current.job_id !== '' &&
            current.job_id === status.job_id &&
            TERMINAL_STATUSES.has(current.status) &&
            !TERMINAL_STATUSES.has(status.status)
        ) {
            return map;
        }
        const next = new Map(map);
        next.set(status.dataset_name, status);
        return next;
    });
    if (TERMINAL_STATUSES.has(status.status)) {
        _scheduleEviction(status.dataset_name);
    } else {
        _clearEviction(status.dataset_name);
    }
}

/** Drop the status entry for a dataset. Called by the action functions when
 *  an HTTP request fails before the server ever emits a status event, so the
 *  optimistic 'running' seed doesn't linger. Gated: only fires when the
 *  current entry is 'idle' or 'running' (the optimistic seed the caller just
 *  set). Pass ``force: true`` to bypass the gate. */
export function resetTaggingStatus(datasetName: string, force: boolean = false): void {
    _taggingStatuses.update((map) => {
        const current = map.get(datasetName);
        if (!force) {
            if (current === undefined) {
                return map;
            }
            if (current.status !== 'idle' && current.status !== 'running') {
                return map;
            }
        }
        _clearEviction(datasetName);
        const next = new Map(map);
        next.delete(datasetName);
        return next;
    });
}
