import { readonly, writable, type Readable } from 'svelte/store';
import type { TaggerStatus } from './types';

/** Lifecycle of the tagger subprocess. Single global slot (the tagger is
 *  one lazily-spawned process, not per-dataset). Fed by the SSE
 *  ``tagger_status`` event. ``state === 'stopped'`` is the initial state —
 *  the subprocess isn't spawned until the first request. */
const _taggerStatus = writable<TaggerStatus>({
    state: 'stopped',
    source: '',
    error: null
});

export const taggerStatus: Readable<TaggerStatus> = readonly(_taggerStatus);

/** Set the tagger subprocess status. Called by the SSE ``tagger_status``
 *  handler. */
export function setTaggerStatus(status: TaggerStatus): void {
    _taggerStatus.set(status);
}
