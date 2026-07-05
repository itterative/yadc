/** Per-dataset tag policy: always-add / banned lists applied during tag result filtering.
 *
 *  The lists live on the backend per dataset under
 *  ``/api/datasets/<name>/tag/policy`` (stored in the server-side
 *  ``dataset_settings`` table as the ``policy_always_add`` /
 *  ``policy_banned`` rows). This module is the **mirror store**:
 *  reactive in-memory copy used by :comp:`PolicyList` / :comp:`Tags.svelte`
 *  for UI reactivity, plus mutators that PUT optimistically against
 *  the dataset that ``loadTagPolicy`` last populated.
 *
 *  Lifecycle:
 *
 *  - **One fetch per dataset**: :func:`loadTagPolicy(datasetName)`
 *    populates the store from the server when the active dataset
 *    changes (page-level effect). A repeat call for the same dataset
 *    is a no-op (returns the cached promise).
 *  - **Optimistic mutations**: every mutator updates the store first
 *    and PUTs the full policy. A failure reverts and rethrows.
 *  - **Dataset switch**: when the page switches to a different
 *    dataset, call :func:`loadTagPolicy(newName)` again — the store
 *    resets to that dataset's persisted policy. The previous dataset's
 *    in-memory state is dropped; mutators targeting the old name
 *    would fail (the PUT endpoint returns 404) and the component is
 *    expected to gate on ``$tagPolicy.datasetName``.
 */

import { derived, get, writable } from 'svelte/store';
import { fetchTagPolicy, putTagPolicy, type TagPolicyPayload } from './api';

/** Mirror-store state — same payload as the backend ``TagPolicy``
 *  dataclass plus a ``datasetName`` discriminator so the UI can refuse
 *  mutations against a dataset that isn't currently loaded. Camel-case
 *  locally because the rest of the tagging stores use it; the wire
 *  adapter below translates to ``always_add`` / ``banned``. */
export interface TagPolicyMirror {
    /** The dataset this policy belongs to. ``null`` until the first
     *  :func:`loadTagPolicy` resolves. Mutators refuse to fire when
     *  this is ``null`` (or, defensively, when it doesn't match the
     *  most recently loaded dataset). */
    datasetName: string | null;
    alwaysAdd: string[];
    banned: string[];
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

/** Single-flight cache: a pending :func:`loadTagPolicy(datasetName)` is
 *  shared across concurrent callers (page mount + a settings panel
 *  binding both firing on the same route change). */
const _loadPromises = new Map<string, Promise<TagPolicyMirror>>();

/** Replace ``tagPolicy`` with the persisted state for ``datasetName``.
 *
 *  Idempotent for a repeated call against the same dataset — the
 *  existing in-flight promise (or the already-loaded value) is
 *  reused. Switching to a different dataset refetches automatically;
 *  the previous dataset's mirror is dropped (its policy still lives
 *  on the server, just not in memory).
 *
 *  Throws on a non-200 (e.g. unregistered dataset → 404) so the
 *  caller can toast the failure; the mirror stays at the previous
 *  state on error. */
export async function loadTagPolicy(datasetName: string): Promise<TagPolicyMirror> {
    const existing = _loadPromises.get(datasetName);
    if (existing !== undefined) {
        return existing;
    }
    const promise = (async () => {
        const payload = await fetchTagPolicy(datasetName);
        const value: TagPolicyMirror = {
            datasetName,
            alwaysAdd: [...payload.always_add],
            banned: [...payload.banned]
        };
        tagPolicy.set(value);
        return value;
    })();
    _loadPromises.set(datasetName, promise);
    try {
        return await promise;
    } catch (e) {
        // Don't cache a failed load so the next call retries cleanly.
        _loadPromises.delete(datasetName);
        throw e;
    }
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

async function persist(value: TagPolicyMirror): Promise<void> {
    const datasetName = value.datasetName;
    if (datasetName === null) {
        throw new Error('tagPolicy is not loaded — call loadTagPolicy(datasetName) first');
    }
    const previous = get(tagPolicy);
    tagPolicy.set(value);
    try {
        await putTagPolicy(datasetName, {
            always_add: [...value.alwaysAdd],
            banned: [...value.banned]
        });
    } catch (e) {
        tagPolicy.set(previous);
        throw e;
    }
}

function ensureUnique(list: string[], tag: string): string[] {
    return list.includes(tag) ? list : [...list, tag];
}

function removeOnce(list: string[], tag: string): string[] {
    const idx = list.indexOf(tag);
    if (idx < 0) {
        return list;
    }
    const next = list.slice();
    next.splice(idx, 1);
    return next;
}

/** Add ``tag`` to the always-add list (idempotent). */
export function addToAlwaysAdd(tag: string): Promise<void> {
    ensureLoaded();
    return persist({
        ...get(tagPolicy),
        alwaysAdd: ensureUnique(get(tagPolicy).alwaysAdd, tag)
    });
}

/** Remove ``tag`` from the always-add list. */
export function removeFromAlwaysAdd(tag: string): Promise<void> {
    ensureLoaded();
    return persist({
        ...get(tagPolicy),
        alwaysAdd: removeOnce(get(tagPolicy).alwaysAdd, tag)
    });
}

/** Add ``tag`` to the banned list (idempotent). */
export function addToBanned(tag: string): Promise<void> {
    ensureLoaded();
    return persist({
        ...get(tagPolicy),
        banned: ensureUnique(get(tagPolicy).banned, tag)
    });
}

/** Remove ``tag`` from the banned list. */
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

/* ───── backwards-compatible wire shape ─────
 *
 * :class:`TagPolicyMirror` is the new mirror store shape. Some
 * callers / pre-existing tests still expect the wire-only subset
 * (``always_add`` / ``banned``), so ``snapshotTagPolicy`` keeps that
 * contract. */

export interface TagPolicySnapshot {
    always_add: string[];
    banned: string[];
}

/** Compute a wire snapshot for code that hasn't migrated to the mirror
 *  store shape (the action layer for example, which used to forward
 *  this on every request — now dropped). The backend no longer reads
 *  these on the wire, but the helper stays so the few remaining
 *  consumers (and the legacy tests) keep working. */
export function snapshotTagPolicy(): TagPolicySnapshot {
    const p = get(tagPolicy);
    return { always_add: [...p.alwaysAdd], banned: [...p.banned] };
}

/* Reactive mirrors for components that want to bind a chip row directly.
 * Discouraged for new code: use ``$tagPolicy.alwaysAdd`` / ``$tagPolicy.banned``
 * directly (with a guard for ``datasetName === null``). */

export const alwaysAddList = derived(tagPolicy, ($p) => $p.alwaysAdd);
export const bannedList = derived(tagPolicy, ($p) => $p.banned);

/** Re-export of the wire-shape type for callers that need it (e.g.
 *  the action layer when constructing test fixtures). */
export type { TagPolicyPayload };
