import { writable, readonly, type Readable } from 'svelte/store';

/** Set of dataset names that have pending filesystem changes (not yet
 *  refreshed). Seeded by the ``dataset_changed`` SSE event when the
 *  change didn't come from this tab. */
const _pendingDatasetChanges = writable<Set<string>>(new Set());

/** Public readonly view. */
export const pendingDatasetChanges: Readable<Set<string>> = readonly(_pendingDatasetChanges);

/** Mark a dataset as having pending changes. Called by the SSE
 *  ``dataset_changed`` handler when the change didn't come from this
 *  tab. */
export function addPendingDatasetChange(datasetName: string): void {
    _pendingDatasetChanges.update((set) => {
        const next = new Set(set);
        next.add(datasetName);
        return next;
    });
}

/** Clear the pending-change flag for a dataset (call after the user
 *  refreshes). */
export function clearPendingDatasetChange(datasetName: string): void {
    _pendingDatasetChanges.update((set) => {
        const next = new Set(set);
        next.delete(datasetName);
        return next;
    });
}
