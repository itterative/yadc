import { get, readonly, writable, type Readable } from 'svelte/store';
import type { ImageRefinedEvent } from '../events';

/** Latest refined caption received via SSE. Single-slot — concurrent
 *  refines for the same image+source+draft can clobber each other. */
const _imageRefined = writable<ImageRefinedEvent | null>(null);

export const imageRefined: Readable<ImageRefinedEvent | null> = readonly(_imageRefined);

/** Set the latest refined caption. Called by the SSE ``image_refined``
 *  handler. */
export function setImageRefined(event: ImageRefinedEvent): void {
    _imageRefined.set(event);
}

/** Return the latest refined caption for the given image + source and
 *  clear it. Returns undefined if no matching refine event is pending.
 *
 *  FIXME: This uses a single writable store, so only one refine result
 *  can be buffered at a time. In practice, refine is limited to one
 *  image per dataset at a time (the captioning job is per-dataset), so
 *  concurrent refines for different images in the same dataset are not
 *  possible. Cross-dataset concurrent refines could overwrite each
 *  other's event — low severity but worth noting. */
export function consumeImageRefined(
    imageId: number,
    source: 'caption' | 'draft' = 'caption',
    draftName: string = ''
): string | undefined {
    const event = get(_imageRefined);
    if (
        event &&
        event.image_id === imageId &&
        event.source === source &&
        (source !== 'draft' || event.draft_name === draftName)
    ) {
        _imageRefined.set(null);
        return event.caption;
    }
    return undefined;
}
