/** Display + comparison helpers for tag identities.
 *
 *  The tagging stores keep curated entries in **canonical** form
 *  (``speech_bubble``); the prune grid's model output is **display**
 *  form (``speech bubble``, after the backend applies the user's
 *  ``replace_underscores``). These two helpers bridge the two worlds:
 *
 *  - :func:`displayTag` projects a canonical curated entry to the
 *    user's preferred display form (the render concern).
 *  - :func:`normKey` collapses either form to a single comparison key
 *    so dedupe / membership checks work regardless of which side a
 *    string came from (mirrors the backend
 *    ``yadc.taggers.postprocessing._norm_key``).
 */

import type { TaggedEntry } from './highlights';

/** Project a curated tag entry to its display form.
 *
 *  ``canonical_form: true`` means the entry's name is the canonical
 *  (model-output) form — the user's ``replaceUnderscores`` preference
 *  is applied. ``canonical_form: false`` means the name is in display
 *  form already (free-text user input, or a kaomoji whose
 *  underscores must be preserved) — render verbatim. Kaomojis are
 *  marked ``canonical_form: false`` by the backend's
 *  :class:`TaggedEntry` validator, so the frontend never has to
 *  special-case them. */
export function displayTag(entry: TaggedEntry, replaceUnderscores: boolean): string {
    if (!entry.canonical_form) {
        return entry.name;
    }
    return replaceUnderscores ? entry.name.replace(/_/g, ' ') : entry.name;
}

/** Normalize a tag string to a lossy comparison key.
 *
 *  Trims, lowercases, and collapses any internal whitespace run to a
 *  single ``_`` (``'Speech Bubble'`` / ``'speech bubble'`` /
 *  ``'speech_bubble'`` → ``'speech_bubble'``). Used for dedupe /
 *  membership so a canonical curated entry and a display-form model
 *  tag resolve to the same identity without forcing either side to a
 *  single form. */
export function normKey(tag: string): string {
    return tag.trim().toLowerCase().split(/\s+/).join('_');
}
