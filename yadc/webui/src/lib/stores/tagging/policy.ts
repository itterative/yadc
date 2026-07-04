/** Auto-include / auto-exclude tag policy for batch tagging jobs.

Separate from :mod:`tagSettings` (thresholds + save options) and
:mod:`highlights` (visual highlight tiers — starred/desired/undesired):
the policy represents the **always-add / banned** lists the user wants
the tagger to apply to every image's result, independent of the visual
tiers on the customize panel. The Customize tab keeps curating
highlight tiers unchanged; the Settings tab's new "Tags" section
curates the policy.

Persistence is localStorage-backed (``yadc/tagPolicy``, ``$version 1``)
so it survives reloads the same way the highlight tiers do. Schema is
versioned but started at 1 — grow it via the ``migrate`` argument when
the shape changes, not via the embedded ``$version`` field.
*/

import { derived, get } from 'svelte/store';
import storable from '$lib/storable.js';
import { z } from 'zod';

/** Body shape for the always-add / banned lists. */
export interface TagPolicy {
    $version: number;
    /** Tags to inject into every tagged image's result at score 1.0. */
    alwaysAdd: string[];
    /** Tags to remove from every tagged image's result entirely. */
    banned: string[];
}

const TagPolicySchema = z.object({
    $version: z.number(),
    alwaysAdd: z.array(z.string()).default([]),
    banned: z.array(z.string()).default([])
});

export const tagPolicy = storable(
    'yadc/tagPolicy',
    {
        $version: 1,
        alwaysAdd: [],
        banned: []
    },
    null,
    TagPolicySchema
);

/* ───── mutators ─────
 *
 * Each mutator targets exactly one list, never the other, so a caller
 * can't accidentally corrupt both at once. Membership is binary (in or
 * out); the helpers are idempotent under repeated adds (a tag added
 * twice lands in the list once) and order-insensitive (they don't
 * sort — the backend's Pydantic-coerced list is irrelevant to
 * equality, and the local order is just insertion order, which is
 * what the chip row shows). */

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
export function addToAlwaysAdd(tag: string): void {
    tagPolicy.update((p) => ({ ...p, alwaysAdd: ensureUnique(p.alwaysAdd, tag) }));
}

/** Remove ``tag`` from the always-add list. */
export function removeFromAlwaysAdd(tag: string): void {
    tagPolicy.update((p) => ({ ...p, alwaysAdd: removeOnce(p.alwaysAdd, tag) }));
}

/** Add ``tag`` to the banned list (idempotent). */
export function addToBanned(tag: string): void {
    tagPolicy.update((p) => ({ ...p, banned: ensureUnique(p.banned, tag) }));
}

/** Remove ``tag`` from the banned list. */
export function removeFromBanned(tag: string): void {
    tagPolicy.update((p) => ({ ...p, banned: removeOnce(p.banned, tag) }));
}

/** Drop every entry in the always-add list. */
export function clearAlwaysAdd(): void {
    tagPolicy.update((p) => ({ ...p, alwaysAdd: [] }));
}

/** Drop every entry in the banned list. */
export function clearBanned(): void {
    tagPolicy.update((p) => ({ ...p, banned: [] }));
}

/** Wire-only subset of :ts:def:`TagPolicy` — excludes the ``$version``
 *  storage marker and matches the backend ``TagPolicy`` pydantic
 *  model's serialised shape (``always_add``/``banned`` keys). Used by
 *  the action layer to forward the current policy to the API. */
export interface TagPolicySnapshot {
    alwaysAdd: string[];
    banned: string[];
}

/** Compute a :ts:def:`TagPolicySnapshot` for the action layer to send
 *  with each tagging request. Reads the storable synchronously so the
 *  call site doesn't need to be inside an effect; same shape as the
 *  request body, which makes the wire mapping trivial. */
export function snapshotTagPolicy(): TagPolicySnapshot {
    const p = get(tagPolicy);
    return { alwaysAdd: [...p.alwaysAdd], banned: [...p.banned] };
}

/* Reactive mirrors for components that want to bind a chip row directly. */

export const alwaysAddList = derived(tagPolicy, ($p) => $p.alwaysAdd);
export const bannedList = derived(tagPolicy, ($p) => $p.banned);
