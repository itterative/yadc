import { derived } from 'svelte/store';
import storable from '$lib/storable.js';

/** User-curated tag tiers, used to visually emphasise specific tags in the
 *  prune grid. Persisted globally (localStorage) for now — tag names only,
 *  no per-dataset or backend binding yet. A tag lives in at most one tier. */
export type TagTier = 'starred' | 'desired' | 'undesired';

/** Named tier values so callers avoid magic strings — ``t === TAG_TIERS.starred``
 *  is IDE-navigable and refactor-safe. */
export const TAG_TIERS = {
    starred: 'starred',
    desired: 'desired',
    undesired: 'undesired'
} as const satisfies Record<TagTier, TagTier>;

export interface TagHighlights {
    $version: number;
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

export const tagHighlights = storable<TagHighlights>('yadc/tagHighlights', {
    $version: 1,
    starred: [],
    desired: [],
    undesired: [],
    categoryOverrides: {}
});

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

/** Assign a tag to a tier, removing it from any other tier first (tiers are
 *  mutually exclusive). Idempotent if the tag is already in that tier. */
export function setTagTier(tier: TagTier, tag: string) {
    tagHighlights.update((h) => withoutTag({ ...h }, tag, tier));
}

/** Remove a tag from every tier. */
export function removeTagTier(tag: string) {
    tagHighlights.update((h) => withoutTag({ ...h }, tag));
}

/** Empty a single tier. */
export function clearTier(tier: TagTier) {
    tagHighlights.update((h) => ({ ...h, [TIER_KEYS[tier]]: [] }));
}

/** Force a starred tag into a specific section (``character`` / ``general``),
 *  overriding the model-derived destination. */
export function setTagCategoryOverride(tag: string, category: string) {
    tagHighlights.update((h) => ({
        ...h,
        categoryOverrides: { ...h.categoryOverrides, [tag]: category }
    }));
}

/** Clear a starred tag's section override (revert to model-derived). */
export function removeTagCategoryOverride(tag: string) {
    tagHighlights.update((h) => {
        const next = { ...h.categoryOverrides };
        delete next[tag];
        return { ...h, categoryOverrides: next };
    });
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
