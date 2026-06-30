import storable from '$lib/storable.js';

/** Recently-added tag, persisted to localStorage so the user can reuse tags
 *  they've added across images. ``category`` is the catalog category the tag
 *  was picked with, or ``'custom'`` for a tag the user typed manually. */
export interface RecentTag {
    name: string;
    category: string;
}

/** Maximum entries kept in localStorage. The dropdown only shows what fits in
 *  its scroll viewport anyway; 100 is a generous ceiling that keeps the
 *  persisted payload small. */
export const RECENT_TAGS_MAX = 100;

export interface RecentTagsData {
    $version: number;
    tags: RecentTag[];
}

export const recentTags = storable<RecentTagsData>(
    'yadc/recentTags',
    { $version: 1, tags: [] },
    null
);

/** Record (or refresh) a recently-added tag. Dedupes by name — the tag moves
 *  to the front so the list reflects recency, not first-seen order. Cap keeps
 *  the persisted payload bounded. Safe to call for tags already present. */
export function recordRecentTag(name: string, category: string): void {
    const trimmed = name.trim();
    if (!trimmed) {
        return;
    }
    recentTags.update((data) => {
        const rest = data.tags.filter((t) => t.name !== trimmed);
        rest.unshift({ name: trimmed, category });
        if (rest.length > RECENT_TAGS_MAX) {
            rest.length = RECENT_TAGS_MAX;
        }
        return { ...data, tags: rest };
    });
}

/** Drop a tag from recent history (the dropdown's per-row × button). No-op if
 *  the name isn't present. */
export function removeRecentTag(name: string): void {
    recentTags.update((data) => ({
        ...data,
        tags: data.tags.filter((t) => t.name !== name)
    }));
}

export function clearRecentTags(): void {
    recentTags.update((data) => ({ ...data, tags: [] }));
}
