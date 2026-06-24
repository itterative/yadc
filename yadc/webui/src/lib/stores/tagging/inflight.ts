import { derived, writable, type Readable, type Writable } from 'svelte/store';

/** Pair identifying an image currently being tagged. */
export type TaggingTarget = { dataset_name: string; image_id: number };

function key(dataset_name: string, image_id: number): string {
    return `${dataset_name}#${image_id}`;
}

/** Images currently being tagged, across all datasets. Empty when idle.
 *  Entries are added optimistically (action) / on ``image_tagged``'s
 *  predecessor and removed on ``image_tagged`` / ``image_tag_error``.
 *  Entries for a dataset are also cleared when its job reaches a terminal
 *  state, catching cancelled in-flight images whose per-image events never
 *  arrived. Internally a ``Map`` keyed by composite string (so the SSE
 *  removal handler reliably matches the optimistic insert). */
const _currentlyTaggingMap: Writable<ReadonlyMap<string, TaggingTarget>> = writable(new Map());

export const currentlyTagging: Readable<ReadonlySet<TaggingTarget>> = derived(
    _currentlyTaggingMap,
    ($map) => new Set($map.values())
);

export function addCurrentlyTagging(datasetName: string, imageId: number): void {
    const k = key(datasetName, imageId);
    _currentlyTaggingMap.update((map) => {
        if (map.has(k)) {
            return map;
        }
        const next = new Map(map);
        next.set(k, { dataset_name: datasetName, image_id: imageId });
        return next;
    });
}

export function removeCurrentlyTagging(datasetName: string, imageId: number): void {
    const k = key(datasetName, imageId);
    _currentlyTaggingMap.update((map) => {
        if (!map.has(k)) {
            return map;
        }
        const next = new Map(map);
        next.delete(k);
        return next;
    });
}

/** Remove all in-flight entries for a dataset (terminal job state). */
export function clearCurrentlyTagging(datasetName: string): void {
    _currentlyTaggingMap.update((map) => {
        let changed = false;
        const next = new Map(map);
        const prefix = `${datasetName}#`;
        for (const k of map.keys()) {
            if (k.startsWith(prefix)) {
                next.delete(k);
                changed = true;
            }
        }
        return changed ? next : map;
    });
}
