/** Per-dataset tag policy: always-add / banned lists applied during tag result filtering.
 *
 *  The lists live on the backend per dataset under
 *  ``/api/datasets/<name>/tag/policy`` (stored in the server-side
 *  ``dataset_settings`` table as the ``policy_always_add`` /
 *  ``policy_banned`` rows). This module is the **mirror store**:
 *  reactive in-memory copy used by :comp:`PolicyList` /
 *  :comp:`Tags.svelte` for UI reactivity, plus mutators that PUT
 *  optimistically against the dataset that :func:`loadTagPolicy` last
 *  populated.
 *
 *  Lifecycle:
 *
 *  - **One fetch per dataset switch**: :func:`loadTagPolicy(datasetName)`
 *    is called from a page-level effect (e.g. in
 *    :comp:`TagSettingsPanel` and :comp:`Tags.svelte`) whenever the
 *    active dataset changes.
 *  - **Optimistic mutations**: every mutator updates the local
 *    mirror first and PUTs the full payload against the active
 *    dataset. A PUT failure reverts and rethrows.
 *  - **Dataset switch**: when the page switches to a different
 *    dataset, call :func:`loadTagPolicy(newName)` again — the
 *    mirror resets to that dataset's persisted policy. The previous
 *    dataset's in-memory state is dropped; mutators targeting the
 *    old name would fail (the PUT endpoint returns 404) and the
 *    component is expected to gate on ``$tagPolicy.datasetName``.
 *
 *  The wire / mirror shape is per-entry ``{name, canonical_form}``,
 *  with ``name`` in **canonical** form. ``canonical_form: true``
 *  means the entry's name is the canonical (model-output) form —
 *  the frontend applies the user's ``replaceUnderscores`` preference
 *  at render time. ``false`` means the name is in display form
 *  already (free-text user input, or a kaomoji) — render verbatim.
 */

import { derived, get, writable } from 'svelte/store';
import { debounce } from '$lib/async';
import { fetchTagPolicy, putTagPolicy, type TagPolicyPayload } from './api';
import { type TaggedEntry } from './highlights';

/** Mirror-store state — same payload as the backend ``StoredPolicy``
 *  (per-entry) plus a ``datasetName`` discriminator so the UI can
 *  refuse mutations against a dataset that isn't currently loaded. */
export interface TagPolicyMirror {
    /** The dataset this policy belongs to. ``null`` until the first
     *  :func:`loadTagPolicy` resolves. Mutators refuse to fire when
     *  this is ``null`` (or, defensively, when it doesn't match the
     *  most recently loaded dataset). */
    datasetName: string | null;
    alwaysAdd: TaggedEntry[];
    banned: TaggedEntry[];
}

const DEFAULT_POLICY: TagPolicyMirror = {
    datasetName: null,
    alwaysAdd: [],
    banned: []
};

export const tagPolicy = writable<TagPolicyMirror>({
    ...DEFAULT_POLICY,
    alwaysAdd: [...DEFAULT_POLICY.alwaysAdd],
    banned: [...DEFAULT_POLICY.banned]
});

/** Populate ``tagPolicy`` with the persisted state for ``datasetName``,
 *  replacing any previous dataset's mirror. */
export const loadTagPolicy = debounce(_loadTagPolicy);

async function _loadTagPolicy(datasetName: string): Promise<TagPolicyMirror> {
    const payload = await fetchTagPolicy(datasetName);
    const value: TagPolicyMirror = {
        datasetName,
        alwaysAdd: [...payload.always_add],
        banned: [...payload.banned]
    };
    tagPolicy.set(value);
    return value;
}

/* ───── mutators ─────
 *
 * Each mutator targets exactly one list, never the other, so a caller
 * can't accidentally corrupt both at once. Membership is binary (in or
 * out); the helpers are idempotent under repeated adds (a tag added
 * twice lands in the list once) and order-insensitive (they don't
 * sort — the wire shape isn't insertion-order-significant).
 *
 *  All mutators check :attr:`TagPolicyMirror.datasetName` against the
 *  current store value: if the store isn't loaded yet (``null``) or
 *  doesn't match the dataset a caller is targeting, the call is a
 *  no-op that returns a rejected promise. Components are expected to
 *  call :func:`loadTagPolicy` before any mutation. */

function ensureLoaded(): string {
    const current = get(tagPolicy);
    if (current.datasetName === null) {
        throw new Error('tagPolicy is not loaded — call loadTagPolicy(datasetName) first');
    }
    return current.datasetName;
}

function payloadFor(value: TagPolicyMirror): TagPolicyPayload {
    return {
        always_add: [...value.alwaysAdd],
        banned: [...value.banned]
    };
}

async function persist(value: TagPolicyMirror): Promise<void> {
    const datasetName = value.datasetName;
    if (datasetName === null) {
        throw new Error('tagPolicy is not loaded — call loadTagPolicy(datasetName) first');
    }
    const previous = get(tagPolicy);
    tagPolicy.set(value);
    try {
        const response = await putTagPolicy(datasetName, payloadFor(value));
        tagPolicy.set(policyToMirror(response, datasetName));
    } catch (e) {
        tagPolicy.set(previous);
        throw e;
    }
}

function ensureUnique(list: TaggedEntry[], entry: TaggedEntry): TaggedEntry[] {
    return list.some((e) => e.name === entry.name) ? list : [...list, entry];
}

function removeOnce(list: TaggedEntry[], name: string): TaggedEntry[] {
    const idx = list.findIndex((e) => e.name === name);
    if (idx < 0) {
        return list;
    }
    const next = list.slice();
    next.splice(idx, 1);
    return next;
}

/** Add ``entry`` to the always-add list (idempotent). ``entry`` is the
 *  canonical ``{name, canonical_form}`` identity, stored verbatim. */
export function addToAlwaysAdd(entry: TaggedEntry): Promise<void> {
    ensureLoaded();
    return persist({
        ...get(tagPolicy),
        alwaysAdd: ensureUnique(get(tagPolicy).alwaysAdd, entry)
    });
}

/** Remove ``tag`` from the always-add list. ``tag`` is the canonical
 *  ``name`` (the chip's ``entry.name``), matched against the mirror. */
export function removeFromAlwaysAdd(tag: string): Promise<void> {
    ensureLoaded();
    return persist({
        ...get(tagPolicy),
        alwaysAdd: removeOnce(get(tagPolicy).alwaysAdd, tag)
    });
}

/** Add ``entry`` to the banned list (idempotent). ``entry`` is the
 *  canonical ``{name, canonical_form}`` identity, stored verbatim. */
export function addToBanned(entry: TaggedEntry): Promise<void> {
    ensureLoaded();
    return persist({
        ...get(tagPolicy),
        banned: ensureUnique(get(tagPolicy).banned, entry)
    });
}

/** Remove ``tag`` from the banned list. ``tag`` is the canonical
 *  ``name`` (the chip's ``entry.name``), matched against the mirror. */
export function removeFromBanned(tag: string): Promise<void> {
    ensureLoaded();
    return persist({
        ...get(tagPolicy),
        banned: removeOnce(get(tagPolicy).banned, tag)
    });
}

/** Drop every entry in the always-add list. */
export function clearAlwaysAdd(): Promise<void> {
    ensureLoaded();
    return persist({ ...get(tagPolicy), alwaysAdd: [] });
}

/** Drop every entry in the banned list. */
export function clearBanned(): Promise<void> {
    ensureLoaded();
    return persist({ ...get(tagPolicy), banned: [] });
}

function policyToMirror(payload: TagPolicyPayload, datasetName: string): TagPolicyMirror {
    return {
        datasetName,
        alwaysAdd: [...payload.always_add],
        banned: [...payload.banned]
    };
}

/* Reactive mirrors for components that want to bind a chip row directly.
 * Discouraged for new code: use ``$tagPolicy.alwaysAdd`` / ``$tagPolicy.banned``
 * directly (with a guard for ``datasetName === null``). */
export const alwaysAddList = derived(tagPolicy, ($p) => $p.alwaysAdd);
export const bannedList = derived(tagPolicy, ($p) => $p.banned);

/** Re-export of the wire-shape type for callers that need it (e.g.
 *  the action layer when constructing test fixtures). */
export type { TagPolicyPayload };
