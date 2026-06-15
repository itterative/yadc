import { derived, writable, type Readable, type Writable } from 'svelte/store';

/** Pair identifying an image currently being captioned. */
export type CaptioningTarget = { dataset_name: string; image_id: number };

/** Composite key for the in-flight map. */
function key(dataset_name: string, image_id: number): string {
    return `${dataset_name}#${image_id}`;
}

/** Images currently being captioned, across all datasets. Empty when
 *  idle. Holds at most ``max_concurrent`` entries per dataset — one
 *  per in-flight parallel task. Entries are added on
 *  ``image_caption_started`` and removed on ``image_caption_captioned``
 *  or ``image_caption_error``. Entries for a dataset are also cleared
 *  when the job for that dataset reaches a terminal state (done /
 *  error / cancelled), which catches images whose per-image events
 *  were never delivered (e.g. cancelled in-flight tasks).
 *
 *  Internally a ``Map`` keyed by ````${dataset_name}#${image_id}````
 *  so the SSE ``image_caption_captioned`` handler can reliably remove
 *  the entry that ``image_caption_started`` added (plain ``Set`` would
 *  fail because every fresh object literal has a different identity).
 *  The public store exposes a ``Set`` of the same entries. */
const _currentlyCaptioningMap: Writable<ReadonlyMap<string, CaptioningTarget>> = writable(
    new Map()
);

/** Public set view of the in-flight images. */
export const currentlyCaptioning: Readable<ReadonlySet<CaptioningTarget>> = derived(
    _currentlyCaptioningMap,
    ($map) => new Set($map.values())
);

/** Add an image to the in-flight set (idempotent). Used to seed
 *  immediate UI feedback before the SSE ``image_caption_started``
 *  event arrives. */
export function addCurrentlyCaptioning(datasetName: string, imageId: number): void {
    const k = key(datasetName, imageId);
    _currentlyCaptioningMap.update((map) => {
        if (map.has(k)) {
            return map;
        }
        const next = new Map(map);
        next.set(k, { dataset_name: datasetName, image_id: imageId });
        return next;
    });
}

/** Remove a single image from the in-flight set (idempotent). Called
 *  from the SSE ``image_captioned`` / ``image_caption_error`` /
 *  ``image_refined`` handlers AND from the action functions when an
 *  HTTP request fails before the server ever emits
 *  ``image_caption_started``. */
export function removeCurrentlyCaptioning(datasetName: string, imageId: number): void {
    const k = key(datasetName, imageId);
    _currentlyCaptioningMap.update((map) => {
        if (!map.has(k)) {
            return map;
        }
        const next = new Map(map);
        next.delete(k);
        return next;
    });
}

/** Remove all in-flight entries for a dataset. Called when a job
 *  reaches a terminal state so cancelled in-flight tasks don't leave
 *  stale entries behind. */
export function clearCurrentlyCaptioning(datasetName: string): void {
    _currentlyCaptioningMap.update((map) => {
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
