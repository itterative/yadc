/** User-curated tag tiers, used to visually emphasise specific tags in the
 *  prune grid. Persisted globally to the backend under
 *  ``/api/tagging/highlights`` (stored in the server-side ``settings``
 *  KV table at ``tagger.tag_highlights``). A tag lives in at most one tier.
 *
 *  The previous shape held a ``$version`` storage marker for the
 *  client-side ``storable`` primitive. The backend doesn't carry
 *  that marker — the wire schema is the versioned contract — so the
 *  marker is dropped here too.
 *
 *  Lifecycle:
 *
 *  - **One-shot fetch on mount**: :func:`ensureHighlightsLoaded` fires
 *    a single ``fetchTagHighlights`` the first time it's called and
 *    seeds the local store. Subsequent calls return the cached promise
 *    so multiple subscribers don't refetch on app boot.
 *  - **Optimistic mutations**: the mutators below update the local
 *    store first, then PUT the full payload. A PUT failure reverts the
 *    local state and rethrows so the caller (action layer / component)
 *    can toast.
 *  - **Refetch**: a hard reload is ``refreshHighlights()`` for the
 *    post-PUT sync path that wants the canonical server state.
 */

import { derived, get, writable } from 'svelte/store';
import { fetchTagHighlights, putTagHighlights, type TagHighlightsPayload } from './api';

export type TagTier = 'starred' | 'desired' | 'undesired';

/** Named tier values so callers avoid magic strings — ``t === TAG_TIERS.starred``
 *  is IDE-navigable and refactor-safe. */
export const TAG_TIERS = {
    starred: 'starred',
    desired: 'desired',
    undesired: 'undesired'
} as const satisfies Record<TagTier, TagTier>;

export interface TagHighlights {
    starred: string[];
    desired: string[];
    undesired: string[];
    /** Starred tag → forced section override (e.g. ``character`` / ``general``).
     *  Absent entry = derive from model output (the default). Only
     *  meaningful for starred tags. */
    categoryOverrides: Record<string, string>;
}

/** The storage key for each tier's tag list. Narrowed to the three
 *  tier-array keys so it can't accidentally widen to other ``TagHighlights``
 *  fields (e.g. ``categoryOverrides``). */
export type TierKey = 'starred' | 'desired' | 'undesired';

export const TIER_KEYS: Readonly<Record<TagTier, TierKey>> = {
    starred: 'starred',
    desired: 'desired',
    undesired: 'undesired'
};

/** Default value used when the store hasn't been populated yet (and
 *  by ``clearTier`` / ``setTagTier`` when constructing a new tier entry).
 *  Stable identity so ``derived`` stores don't churn on every mutation. */
const DEFAULT_HIGHLIGHTS: TagHighlights = {
    starred: [],
    desired: [],
    undesired: [],
    categoryOverrides: {}
};

export const tagHighlights = writable<TagHighlights>({
    ...DEFAULT_HIGHLIGHTS,
    categoryOverrides: { ...DEFAULT_HIGHLIGHTS.categoryOverrides }
});

/** Module-level single-flight promise so multiple ``ensureHighlightsLoaded()``
 *  callers (e.g. multiple route components mounting in parallel) share
 *  one in-flight fetch. Mutated atomically with the store so a
 *  subsequent call after the first resolves doesn't refetch. */
let _loadPromise: Promise<TagHighlights> | null = null;
let _loaded = false;

/** Populate ``tagHighlights`` from the server. Idempotent — the second
 *  call after the first resolves is a no-op. Returns the populated
 *  value. Throws on a non-200 so the caller can toast the failure;
 *  the in-memory store stays at its current value on error. */
export async function ensureHighlightsLoaded(): Promise<TagHighlights> {
    if (_loaded) {
        return get(tagHighlights);
    }
    if (_loadPromise === null) {
        _loadPromise = (async () => {
            const payload = await fetchTagHighlights();
            const value = payloadToHighlights(payload);
            tagHighlights.set(value);
            _loaded = true;
            return value;
        })();
    }
    return _loadPromise;
}

/** Force-refresh from the server. Used by the action layer after a
 *  long pause when a session may have drifted (or in tests). */
export async function refreshHighlights(): Promise<TagHighlights> {
    _loadPromise = null;
    _loaded = false;
    return ensureHighlightsLoaded();
}

/** Reactive ``tag → tier`` lookup. Re-derives whenever the store changes so
 *  chip rendering in the prune grid stays in sync with the customize tab. */
export const tagTierMap = derived(tagHighlights, ($h) => {
    const map = new Map<string, TagTier>();
    for (const t of $h.starred) {
        map.set(t, TAG_TIERS.starred);
    }
    for (const t of $h.desired) {
        map.set(t, TAG_TIERS.desired);
    }
    for (const t of $h.undesired) {
        map.set(t, TAG_TIERS.undesired);
    }
    return map;
});

/** Reactive ``tag → forced section`` lookup for starred tags. Absent =
 *  derive from model output. */
export const tagCategoryOverrideMap = derived(tagHighlights, ($h) => {
    const map = new Map<string, string>();
    for (const [tag, cat] of Object.entries($h.categoryOverrides)) {
        map.set(tag, cat);
    }
    return map;
});

/* ───── mutators ─────
 *
 * Every mutator applies the change locally first (optimistic) and
 * PUTs the full payload to the backend. A PUT failure reverts to the
 * snapshot taken before the local update, then rethrows so the
 * caller (typically the action layer or a component error
 * boundary) can surface a toast. Server is authoritative — on
 * reconnect / refresh the canonical state comes back through
 * :func:`refreshHighlights`. */

async function persist(value: TagHighlights): Promise<void> {
    const previous = get(tagHighlights);
    tagHighlights.set(value);
    try {
        await putTagHighlights(highlightsToPayload(value));
    } catch (e) {
        tagHighlights.set(previous);
        throw e;
    }
}

/** Assign a tag to a tier, removing it from any other tier first (tiers are
 *  mutually exclusive). Idempotent if the tag is already in that tier. */
export function setTagTier(tier: TagTier, tag: string): Promise<void> {
    return persist(withoutTag({ ...get(tagHighlights) }, tag, tier));
}

/** Remove a tag from every tier. */
export function removeTagTier(tag: string): Promise<void> {
    return persist(withoutTag({ ...get(tagHighlights) }, tag));
}

/** Empty a single tier. */
export function clearTier(tier: TagTier): Promise<void> {
    const next: TagHighlights = { ...get(tagHighlights), [TIER_KEYS[tier]]: [] };
    return persist(next);
}

/** Force a starred tag into a specific section (``character`` / ``general``),
 *  overriding the model-derived destination. */
export function setTagCategoryOverride(tag: string, category: string): Promise<void> {
    const next: TagHighlights = {
        ...get(tagHighlights),
        categoryOverrides: { ...get(tagHighlights).categoryOverrides, [tag]: category }
    };
    return persist(next);
}

/** Clear a starred tag's section override (revert to model-derived). */
export function removeTagCategoryOverride(tag: string): Promise<void> {
    const overrides = { ...get(tagHighlights).categoryOverrides };
    delete overrides[tag];
    return persist({ ...get(tagHighlights), categoryOverrides: overrides });
}

function withoutTag(h: TagHighlights, tag: string, except?: TagTier): TagHighlights {
    (Object.keys(TIER_KEYS) as TagTier[]).forEach((t) => {
        if (t === except) {
            return;
        }
        const key = TIER_KEYS[t];
        if (h[key].includes(tag)) {
            h[key] = h[key].filter((x) => x !== tag);
        }
    });
    if (except) {
        const key = TIER_KEYS[except];
        if (!h[key].includes(tag)) {
            h[key] = [...h[key], tag];
        }
    } else {
        // Leaving every tier — drop any section override too (it only
        // applied while the tag was starred).
        if (tag in h.categoryOverrides) {
            const next = { ...h.categoryOverrides };
            delete next[tag];
            h.categoryOverrides = next;
        }
    }
    return h;
}

/* ───── wire helpers ─────
 *
 * Round-trip between the backend wire shape (snake_case field names)
 * and the local store shape (camelCase). Kept private so callers
 * don't depend on the wire format directly. */

function payloadToHighlights(payload: TagHighlightsPayload): TagHighlights {
    return {
        starred: [...payload.starred],
        desired: [...payload.desired],
        undesired: [...payload.undesired],
        categoryOverrides: { ...payload.category_overrides }
    };
}

function highlightsToPayload(value: TagHighlights): TagHighlightsPayload {
    return {
        starred: [...value.starred],
        desired: [...value.desired],
        undesired: [...value.undesired],
        category_overrides: { ...value.categoryOverrides }
    };
}
