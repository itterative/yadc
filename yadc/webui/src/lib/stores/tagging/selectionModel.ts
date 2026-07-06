/** Pure selection-model logic for the per-image tag prune grid. Extracted
 *  from :file:`Tags.svelte` so the fiddly routing / persistence rules are
 *  unit-testable and live in one place rather than being re-implemented
 *  (and silently drifting) across display, save, and snapshot builders.
 *
 *  Three concerns share the same data walk — display ordering
 *  (:func:`computeOrderedCategories`), the saved pruned result, and the
 *  persisted customization snapshot — so the starred-routing rule
 *  (:func:`starredSection`) and the merged builder
 *  (:func:`computeSelection`) are defined once here. */

import { TAG_CATEGORIES } from './constants';
import { TAG_TIERS, type TagTier } from './highlights';
import type { TagCustomizations, TaggerResult } from './types';

/** The two sections a starred tag can be routed into. (``rating`` and any
 *  exotic model category collapse into ``general``.) */
export type TagSection = typeof TAG_CATEGORIES.character | typeof TAG_CATEGORIES.general;

/** One rendered chip — score + flags computed by the parent so the chip
 *  component stays focused on presentation.
 *
 *  ``canonicalForm`` mirrors the curated-tier ``canonical_form`` flag:
 *  ``true`` for catalog / model-output entries (the chip view applies
 *  the user's ``replaceUnderscores`` preference when rendering);
 *  ``false`` for free-text user additions or kaomojis (rendered
 *  verbatim). The chip's :comp:`TagChip` projects ``tag`` through
 *  :func:`displayTag` using this flag. */
export interface TagChipView {
    tag: string;
    score: number;
    canonicalForm: boolean;
    /** Starred tag not detected in this image (force-shown, no score).
     *  Renders without a confidence % and defaults to off. */
    absent?: boolean;
}

export interface OrderedCategory {
    name: string;
    chips: TagChipView[];
}

/** Stable section display order (wd-tagger convention: rating / character /
 *  general); any other model categories sort after, alphabetically. */
export const SECTION_ORDER: readonly string[] = [
    TAG_CATEGORIES.rating,
    TAG_CATEGORIES.character,
    TAG_CATEGORIES.general
];

/** Section a starred tag belongs in. Explicit override wins; otherwise a
 *  character model tag stays in character and everything else (general,
 *  rating, exotic categories, or unknown) collapses to general. Defined
 *  once so display, save, and persistence all agree — drifting here is
 *  exactly how the grid can show a tag under one section while the saved
 *  output files it under another. */
export function starredSection(
    tag: string,
    fallbackCategory: string | undefined,
    overrides: ReadonlyMap<string, string>
): string {
    const ov = overrides.get(tag);
    if (ov === TAG_CATEGORIES.character || ov === TAG_CATEGORIES.general) {
        return ov;
    }
    return fallbackCategory === TAG_CATEGORIES.character
        ? TAG_CATEGORIES.character
        : TAG_CATEGORIES.general;
}

/** Whether ``tag`` is in the starred tier. */
export function isStarredTag(tag: string, tiers: ReadonlyMap<string, TagTier>): boolean {
    return tiers.get(tag) === TAG_TIERS.starred;
}

/** Append to a ``Record<string, string[]>`` bucket, creating the array on
 *  first use. Plain-object category maps aren't pre-seeded with empty
 *  arrays, so a bare ``obj[key].push(...)`` would throw on the first tag
 *  into a new section. */
export function pushBucket(buckets: Record<string, string[]>, key: string, value: string): void {
    if (!buckets[key]) {
        buckets[key] = [];
    }
    buckets[key].push(value);
}

export interface SelectionSnapshot {
    pruned: TaggerResult;
    customizations: TagCustomizations;
}

/** Compute both the saved pruned result and the persisted customization
 *  snapshot in one pass. Walking the data once (instead of twice, in
 *  ``buildPrunedResult`` + ``buildCustomizationsSnapshot``) guarantees the
 *  two outputs can't disagree: a tag is enabled in the save iff it's
 *  enabled in the snapshot, and every starred tag is routed via the same
 *  :func:`starredSection` call.
 *
 *  - Model tags route to their model category, or :func:`starredSection`
 *    when starred. Turned-off model tags are recorded in ``disabled``.
 *  - Custom (user-typed) tags carry a synthetic 1.0 score and their chosen
 *    category (or the override destination when starred).
 *  - Force-enabled absent starred tags (in the tier but not in the model
 *    output or custom map) are saved under their override section at 1.0
 *    and folded into ``custom_tags`` so they survive navigation. */
export function computeSelection(
    result: TaggerResult,
    enabled: ReadonlySet<string>,
    customTags: ReadonlyMap<string, string>,
    tiers: ReadonlyMap<string, TagTier>,
    overrides: ReadonlyMap<string, string>
): SelectionSnapshot {
    const prunedCategories: Record<string, string[]> = {};
    const prunedTags: Record<string, number> = {};
    const disabled: string[] = [];
    const customByCat: Record<string, string[]> = {};

    // Model tags.
    for (const [cat, tags] of Object.entries(result.categories)) {
        for (const t of tags) {
            if (!enabled.has(t)) {
                disabled.push(t);
                continue;
            }
            const dest = isStarredTag(t, tiers) ? starredSection(t, cat, overrides) : cat;
            pushBucket(prunedCategories, dest, t);
            if (t in result.tags) {
                prunedTags[t] = result.tags[t];
            }
        }
    }

    // Custom additions.
    for (const [tag, cat] of customTags) {
        if (!enabled.has(tag)) {
            continue;
        }
        const dest = isStarredTag(tag, tiers) ? starredSection(tag, cat, overrides) : cat;
        pushBucket(prunedCategories, dest, tag);
        pushBucket(customByCat, dest, tag);
        prunedTags[tag] = 1.0;
    }

    // Force-enabled absent starred tags.
    for (const [tag, tier] of tiers) {
        if (tier !== TAG_TIERS.starred || !enabled.has(tag)) {
            continue;
        }
        if (tag in result.tags || customTags.has(tag)) {
            continue;
        }
        const dest = starredSection(tag, undefined, overrides);
        pushBucket(prunedCategories, dest, tag);
        pushBucket(customByCat, dest, tag);
        prunedTags[tag] = 1.0;
    }

    return {
        pruned: { tags: prunedTags, categories: prunedCategories },
        customizations: { disabled, custom_tags: customByCat }
    };
}

/** Build the per-section chip layout for the prune grid. Starred tags are
 *  pulled to the *lead* of their destination section (character or general);
 *  absent starred tags are force-shown as disabled chips so the user can see
 *  which curated picks the model missed. Non-starred tags stay in their
 *  natural category, custom ahead of model by score. */
export function computeOrderedCategories(
    result: TaggerResult,
    customTags: ReadonlyMap<string, string>,
    tiers: ReadonlyMap<string, TagTier>,
    overrides: ReadonlyMap<string, string>
): OrderedCategory[] {
    const isStarred = (t: string) => isStarredTag(t, tiers);

    // Index model tags → detected category for starred-destination fallback.
    const modelCat = new Map<string, string>();
    for (const [cat, tags] of Object.entries(result.categories)) {
        for (const t of tags) {
            modelCat.set(t, cat);
        }
    }

    // Partition starred tags into their destination section's lead block.
    const leadChar: TagChipView[] = [];
    const leadGen: TagChipView[] = [];
    const leadOf = (tag: string) =>
        starredSection(tag, modelCat.get(tag), overrides) === TAG_CATEGORIES.character
            ? leadChar
            : leadGen;

    for (const tags of Object.values(result.categories)) {
        for (const t of tags) {
            if (!isStarred(t)) {
                continue;
            }
            leadOf(t).push({ tag: t, score: result.tags[t] ?? 0, canonicalForm: true });
        }
    }
    for (const [tag] of customTags) {
        if (isStarred(tag)) {
            leadOf(tag).push({ tag, score: 1.0, canonicalForm: false });
        }
    }
    // Absent starred tags (not detected, not custom) — default to general
    // unless an override places them in character.
    const absent: string[] = [];
    for (const [tag, tier] of tiers) {
        if (tier !== TAG_TIERS.starred) {
            continue;
        }
        if (!(tag in result.tags) && !customTags.has(tag)) {
            absent.push(tag);
        }
    }
    absent.sort();
    for (const tag of absent) {
        leadOf(tag).push({ tag, score: 0, canonicalForm: true, absent: true });
    }

    // Sort leads: present by score desc (ties by name), absent last.
    const byScore = (a: TagChipView, b: TagChipView) =>
        b.score - a.score || a.tag.localeCompare(b.tag);
    leadChar.sort(byScore);
    leadGen.sort((a, b) => {
        if (!!a.absent !== !!b.absent) {
            return a.absent ? 1 : -1;
        }
        return byScore(a, b);
    });

    // Non-starred tail (starred pulled into leads above): custom first, then
    // model by score desc.
    const tail = (cat: string): TagChipView[] => {
        const custom: string[] = [];
        const model: string[] = [];
        for (const [tag, c] of customTags) {
            if (c === cat && !isStarred(tag)) {
                custom.push(tag);
            }
        }
        for (const t of result.categories[cat] ?? []) {
            if (!customTags.has(t) && !isStarred(t)) {
                model.push(t);
            }
        }
        custom.sort();
        model.sort((a, b) => (result.tags[b] ?? 0) - (result.tags[a] ?? 0) || a.localeCompare(b));
        return [
            ...custom.map((t) => ({ tag: t, score: 1.0, canonicalForm: false })),
            ...model.map((t) => ({ tag: t, score: result.tags[t] ?? 0, canonicalForm: true }))
        ];
    };

    const sections: OrderedCategory[] = [];
    const seen = new Set<string>();
    for (const name of SECTION_ORDER) {
        seen.add(name);
        if (name === TAG_CATEGORIES.character) {
            const chips = [...leadChar, ...tail(name)];
            if (chips.length > 0 || name in result.categories) {
                sections.push({ name, chips });
            }
        } else if (name === TAG_CATEGORIES.general) {
            const chips = [...leadGen, ...tail(name)];
            // Always render general when it has any lead so absent starred
            // tags show even if the model produced no general tags.
            if (chips.length > 0) {
                sections.push({ name, chips });
            }
        } else {
            const chips = tail(name);
            if (chips.length > 0 || name in result.categories) {
                sections.push({ name, chips });
            }
        }
    }
    const extras = [
        ...Object.keys(result.categories).filter((c) => !seen.has(c)),
        ...[...customTags.values()].filter((c) => !seen.has(c) && !(c in result.categories))
    ].sort();
    for (const name of extras) {
        const chips = tail(name);
        if (chips.length > 0 || name in result.categories) {
            sections.push({ name, chips });
        }
    }
    return sections;
}
