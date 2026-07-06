/** User-curated tag tiers, used to visually emphasise specific tags in the
 *  prune grid. Persisted globally to the backend under
 *  ``/api/tagging/highlights`` (stored in the server-side ``settings``
 *  KV table at ``tagger.tag_highlights``). A tag lives in at most one tier.
 *
 *  The wire / mirror shape is per-entry ``{name, canonical_form}``,
 *  with ``name`` in **canonical** form (``speech_bubble``).
 *  ``canonical_form: true`` means the entry's name is the canonical
 *  (model-output) form and the frontend applies the user's
 *  ``replaceUnderscores`` preference at render time. ``false`` means
 *  the name should be rendered verbatim (free-text user input, or
 *  a kaomoji — the backend's :class:`TaggedEntry` validator
 *  auto-flips the flag for kaomojis). Display is a render concern —
 *  the derived tier / override maps below project canonical keys
 *  to display form so the prune grid (which works in display-form
 *  model tags) keeps matching.
 */

import { derived, get, writable } from 'svelte/store';
import {
    fetchTagHighlights,
    putTagHighlights,
    type TagHighlightsPayload,
    type TaggedEntryPayload
} from './api';
import { displayTag } from './display';
import { tagSettings } from './settings';

export type TagTier = 'starred' | 'desired' | 'undesired';

/** Named tier values so callers avoid magic strings — ``t === TAG_TIERS.starred``
 *  is IDE-navigable and refactor-safe. */
export const TAG_TIERS = {
    starred: 'starred',
    desired: 'desired',
    undesired: 'undesired'
} as const satisfies Record<TagTier, TagTier>;

/** Re-export of the wire entry type for callers that need to
 *  construct one (e.g. mutators, fixtures). ``canonical_form``
 *  defaults to ``true`` to match the backend's permissive default
 *  (catalog / model-output entries are the common case). */
export type TaggedEntry = TaggedEntryPayload;

export interface TagHighlights {
    starred: TaggedEntry[];
    desired: TaggedEntry[];
    undesired: TaggedEntry[];
    /** Starred tag → forced section override (e.g. ``character`` / ``general``).
     *  Absent entry = derive from model output (the default). Only
     *  meaningful for starred tags. */
    categoryOverrides: Record<string, string>;
}

/** The storage key for each tier's entry list. Narrowed to the three
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
    starred: [...DEFAULT_HIGHLIGHTS.starred],
    desired: [...DEFAULT_HIGHLIGHTS.desired],
    undesired: [...DEFAULT_HIGHLIGHTS.undesired],
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

/** Reactive ``display-form tag → tier`` lookup. Keys are projected from
 *  the canonical mirror via the user's ``replaceUnderscores`` preference so
 *  the prune grid (which works in display-form model tags) keeps matching.
 *  Entries with ``canonical_form: false`` (free-text, kaomojis) project
 *  verbatim. Re-derives when the mirror or the setting changes. */
export const tagTierMap = derived([tagHighlights, tagSettings], ([$h, $s]) => {
    const ru = $s.replaceUnderscores;
    const map = new Map<string, TagTier>();
    const add = (entries: TaggedEntry[], tier: TagTier) => {
        for (const e of entries) {
            map.set(displayTag(e, ru), tier);
        }
    };
    add($h.starred, TAG_TIERS.starred);
    add($h.desired, TAG_TIERS.desired);
    add($h.undesired, TAG_TIERS.undesired);
    return map;
});

/** Reactive ``display-form tag → forced section`` lookup for starred tags.
 *  Override keys are stored canonical; this projects them to display form
 *  (using the matching starred entry's ``canonical_form`` flag) so the
 *  prune grid's display-tag lookups resolve. Absent = derive from model
 *  output. */
export const tagCategoryOverrideMap = derived([tagHighlights, tagSettings], ([$h, $s]) => {
    const ru = $s.replaceUnderscores;
    const canonicalByName = new Map($h.starred.map((e) => [e.name, e.canonical_form] as const));
    const map = new Map<string, string>();
    for (const [name, cat] of Object.entries($h.categoryOverrides)) {
        map.set(displayTag({ name, canonical_form: canonicalByName.get(name) ?? true }, ru), cat);
    }
    return map;
});

/* ───── mutators ─────
 *
 * Every mutator updates the store first (optimistic) and
 * PUTs the full payload to the backend. A PUT failure reverts to
 * the snapshot taken before the local update, then rethrows so the
 * caller (typically the action layer or a component error
 * boundary) can surface a toast. */

/** Helper used inside ``persist`` to convert the optimistic-local
 *  tier lists (which may contain entries the previous PUT round
 *  hadn't reached the server yet) into the wire shape. */
function payloadFor(value: TagHighlights): TagHighlightsPayload {
    return {
        starred: [...value.starred],
        desired: [...value.desired],
        undesired: [...value.undesired],
        category_overrides: { ...value.categoryOverrides }
    };
}

async function persist(value: TagHighlights): Promise<void> {
    const previous = get(tagHighlights);
    tagHighlights.set(value);
    try {
        const response = await putTagHighlights(payloadFor(value));
        tagHighlights.set(payloadToHighlights(response));
    } catch (e) {
        tagHighlights.set(previous);
        throw e;
    }
}

/** Assign a tag to a tier, removing it from any other tier first (tiers are
 *  mutually exclusive). Idempotent if the tag is already in that tier.
 *
 *  ``entry`` is the canonical ``{name, canonical_form}`` identity —
 *  stored verbatim; the frontend projects to display at render time. */
export function setTagTier(tier: TagTier, entry: TaggedEntry): Promise<void> {
    return persist(withoutTag({ ...get(tagHighlights) }, entry, tier));
}

/** Remove a tag from every tier. Matches by canonical ``name`` — the
 *  chip's ``entry.name`` is the same shape as the mirror entry's
 *  ``name`` (both canonical), so the iteration is unambiguous. */
export function removeTagTier(tag: string): Promise<void> {
    return persist(withoutTag({ ...get(tagHighlights) }, tag));
}

/** Empty a single tier. */
export function clearTier(tier: TagTier): Promise<void> {
    const next: TagHighlights = { ...get(tagHighlights), [TIER_KEYS[tier]]: [] };
    return persist(next);
}

/** Force a starred tag into a specific section (``character`` / ``general``),
 *  overriding the model-derived destination. ``tag`` is the canonical
 *  name (the TierList chip's ``entry.name``), stored as the override key. */
export function setTagCategoryOverride(tag: string, category: string): Promise<void> {
    const next: TagHighlights = {
        ...get(tagHighlights),
        categoryOverrides: { ...get(tagHighlights).categoryOverrides, [tag]: category }
    };
    return persist(next);
}

/** Clear a starred tag's section override (revert to model-derived). ``tag``
 *  is the canonical name (the TierList chip's ``entry.name``). */
export function removeTagCategoryOverride(tag: string): Promise<void> {
    const overrides = { ...get(tagHighlights).categoryOverrides };
    delete overrides[tag];
    return persist({ ...get(tagHighlights), categoryOverrides: overrides });
}

function withoutTag(
    h: TagHighlights,
    tagOrEntry: string | TaggedEntry,
    except?: TagTier
): TagHighlights {
    const lookupName = typeof tagOrEntry === 'string' ? tagOrEntry : tagOrEntry.name;
    (Object.keys(TIER_KEYS) as TagTier[]).forEach((t) => {
        if (t === except) {
            return;
        }
        const key = TIER_KEYS[t];
        if (h[key].some((e) => e.name === lookupName)) {
            h[key] = h[key].filter((e) => e.name !== lookupName);
        }
    });
    if (except) {
        const key = TIER_KEYS[except];
        if (!h[key].some((e) => e.name === lookupName)) {
            h[key] = [
                ...h[key],
                ...(typeof tagOrEntry === 'string'
                    ? [{ name: tagOrEntry, canonical_form: true }]
                    : [tagOrEntry])
            ];
        }
    } else {
        // Leaving every tier — drop any section override too (it only
        // applied while the tag was starred).
        if (lookupName in h.categoryOverrides) {
            const next = { ...h.categoryOverrides };
            delete next[lookupName];
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
