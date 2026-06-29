import { browser } from '$app/environment';
import { readonly, writable, type Readable, type Writable } from 'svelte/store';
import type { CaptioningStatus } from '../events';

/** A blank idle status — the shape a per-dataset slot starts from. Kept as a
 *  named constant so callers building an optimistic seed don't reinvent it. */
export const INITIAL_CAPTIONING_STATUS: CaptioningStatus = {
    status: 'idle',
    dataset_name: '',
    processed: 0,
    total: 0,
    errors: 0,
    job_id: '',
    error: null,
    error_messages: [],
    elapsed: 0,
    max_concurrent: 1
};

/** Statuses that mean the job is no longer making progress. An entry reaching
 *  one of these is evicted from the map after a grace period so the
 *  "Complete" / "Error" badge flashes on the dataset card and then clears. */
const TERMINAL_STATUSES = new Set<CaptioningStatus['status']>([
    'idle',
    'done',
    'error',
    'cancelled'
]);

/** How long a terminal entry lingers before eviction. The backend supports
 *  captioning several datasets at once; keeping terminal entries around for a
 *  few minutes means the dataset list keeps showing per-dataset "Complete"
 *  results for a batch of jobs that finish around the same time, while still
 *  bounding the map's growth over a long session. */
const TERMINAL_EVICTION_MS = 5 * 60 * 1000;

const _captioningStatuses: Writable<ReadonlyMap<string, CaptioningStatus>> = writable(
    new Map<string, CaptioningStatus>()
);

/** Per-dataset captioning status, keyed by dataset name. Holds one entry per
 *  dataset that is (or recently was) captioning, so the dataset list can show
 *  progress for several concurrent jobs at once instead of just the most
 *  recently reported one. Fed by the SSE ``captioning_status`` event and the
 *  action functions (which seed optimistic state and reset on failure). The
 *  dataset list reads every entry; the detail page / topbar read the entry for
 *  the open dataset via ``.get(name)``. */
export const captioningStatuses: Readable<ReadonlyMap<string, CaptioningStatus>> =
    readonly(_captioningStatuses);

/** Pending eviction timers for terminal entries, keyed by dataset name.
 *  Cleared if a fresh non-terminal status arrives for the same dataset. */
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
            _captioningStatuses.update((map) => {
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

/** Upsert a dataset's status into the map. Called from the SSE
 *  ``captioning_status`` handler and the action functions' optimistic seeds.
 *  Terminal statuses (done/error/cancelled/idle) are scheduled for eviction
 *  after ``TERMINAL_EVICTION_MS``; any pending eviction is cancelled when a
 *  fresh non-terminal status arrives for the same dataset. */
export function setCaptioningStatus(status: CaptioningStatus): void {
    _captioningStatuses.update((map) => {
        const current = map.get(status.dataset_name);
        // Don't overwrite a terminal status from the same job with a
        // non-terminal one. The HTTP response from ``POST /captioner/start``
        // carries a snapshot read at submit time; if the background task
        // finishes (and dispatches the terminal ``captioning_status`` SSE
        // event) before the response lands, a late response that says
        // 'starting'/'running' would otherwise yank the UI back from
        // 'done'. The terminal SSE event is the authoritative state for
        // that job_id; if it never arrives, the eviction timer cleans up
        // after 5 min. (Same gate as ``setTaggingStatus``; captioning
        // doesn't have a tagger-style all-cache fast path, but the race
        // is theoretically possible for small/fast captioning jobs.)
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

/** Drop the status entry for a dataset. Called by the action functions when an
 *  HTTP request fails before the server ever emits a status event, so the
 *  optimistic 'starting' seed doesn't linger.
 *
 *  Gated: only fires when the current entry for ``datasetName`` is 'idle' or
 *  'starting' (the optimistic seed the caller just set). A 'running'/'stopping'
 *  entry is preserved so a second near-simultaneous request that 409s doesn't
 *  clobber the first request's real seed. Pass ``force: true`` to bypass the
 *  gate. */
export function resetCaptioningStatus(datasetName: string, force: boolean = false): void {
    _captioningStatuses.update((map) => {
        const current = map.get(datasetName);
        if (!force) {
            if (current === undefined) {
                return map;
            }
            if (current.status !== 'idle' && current.status !== 'starting') {
                return map;
            }
        }
        _clearEviction(datasetName);
        const next = new Map(map);
        next.delete(datasetName);
        return next;
    });
}
