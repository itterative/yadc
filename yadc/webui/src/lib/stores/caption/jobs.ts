import { get, writable } from 'svelte/store';

/** Job IDs of operations (captioning, tagging, ...) initiated by this
 *  frontend (bounded ring). Used by the SSE ``dataset_changed`` handler
 *  to suppress ``dataset_changed`` events caused by our own jobs (the
 *  backend tags the change with the job_id, and we filter it here). */
const _activeJobIds = writable<string[]>([]);

const MAX_ACTIVE_JOB_IDS = 16;

/** Register a captioning job ID initiated by this frontend. */
export function registerJobId(jobId: string): void {
    _activeJobIds.update((ids) => {
        if (ids.length >= MAX_ACTIVE_JOB_IDS) {
            return [...ids.slice(ids.length - MAX_ACTIVE_JOB_IDS + 1), jobId];
        }
        return [...ids, jobId];
    });
}

/** True when ``jobId`` is one of our own active job IDs. Used by the
 *  SSE ``dataset_changed`` handler to suppress events caused by our
 *  own captioning jobs. */
export function isOwnJobId(jobId: string | null | undefined): boolean {
    if (!jobId) {
        return false;
    }
    return get(_activeJobIds).includes(jobId);
}
