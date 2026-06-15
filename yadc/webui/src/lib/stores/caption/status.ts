import { get, readonly, writable, type Readable, type Writable } from 'svelte/store';
import type { CaptioningStatus } from '../events';

/** Initial idle state — also the target of ``resetCaptioningStatus``. */
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

const _captioningStatus: Writable<CaptioningStatus> = writable(INITIAL_CAPTIONING_STATUS);

/** Latest captioning job status — fed by the SSE ``captioning_status``
 *  event and the action functions (which seed optimistic state and
 *  reset on failure). The topbar reads this. */
export const captioningStatus: Readable<CaptioningStatus> = readonly(_captioningStatus);

/** Explicitly set the status (e.g. from an initial HTTP poll or an
 *  action's optimistic seed). */
export function setCaptioningStatus(status: CaptioningStatus): void {
    _captioningStatus.set(status);
}

/** Reset the captioning status to the initial idle state.
 *
 *  Gated: only fires when the current status is 'idle' or 'starting'
 *  (the optimistic seed the caller just set). A 'running'/'stopping'
 *  state is preserved so a second near-simultaneous request that
 *  409s doesn't clobber the first request's real seed.
 *
 *  Pass ``force: true`` to bypass the gate. */
export function resetCaptioningStatus(force: boolean = false): void {
    if (!force) {
        const current = get(_captioningStatus);
        if (current.status !== 'idle' && current.status !== 'starting') {
            return;
        }
    }
    _captioningStatus.set(INITIAL_CAPTIONING_STATUS);
}
