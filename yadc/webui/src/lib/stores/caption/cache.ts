import { get, readonly, writable, type Readable } from 'svelte/store';

/** Captions received via SSE, keyed by image ID. Bounded LRU — oldest
 *  entries are evicted when the capacity is exceeded. Accessed entries
 *  are promoted so recently-viewed captions survive eviction. */
const _storedCaptions = writable<Map<number, string>>(new Map());

const MAX_STORED_CAPTIONS = 64;

export const storedCaptions: Readable<Map<number, string>> = readonly(_storedCaptions);

/** Return a caption received via SSE for the given image, if any.
 *  Promotes the entry to most-recently-used (LRU eviction ordering). */
export function getStoredCaption(imageId: number): string | undefined {
    const map = get(_storedCaptions);
    const value = map.get(imageId);
    if (value !== undefined) {
        // Promote to end of iteration order (most-recently-used).
        _storedCaptions.update((m) => {
            m.delete(imageId);
            m.set(imageId, value);
            return m;
        });
    }
    return value;
}

/** Remove a stored caption after authoritative data has been fetched. */
export function clearStoredCaption(imageId: number): void {
    _storedCaptions.update((map) => {
        const next = new Map(map);
        next.delete(imageId);
        return next;
    });
}

/** Insert a caption into the LRU cache. Used by the SSE
 *  ``image_captioned`` handler; exported for the cache cap to be
 *  enforced in one place. */
export function putStoredCaption(imageId: number, caption: string): void {
    _storedCaptions.update((map) => {
        const next = new Map(map);
        next.set(imageId, caption);
        if (next.size > MAX_STORED_CAPTIONS) {
            const firstKey = next.keys().next().value;
            if (firstKey !== undefined) {
                next.delete(firstKey);
            }
        }
        return next;
    });
}
