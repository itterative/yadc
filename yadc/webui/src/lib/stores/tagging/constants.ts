/** Named tag-category values (the wd-tagger convention: rating / character /
 *  general). Used across the prune grid, the customize panel, and the save
 *  builder so callers avoid magic strings and get IDE navigation. */
export const TAG_CATEGORIES = {
    rating: 'rating',
    character: 'character',
    general: 'general',
    custom: 'custom'
} as const;

/** Categories that can receive user-added custom tags. ``rating`` is a
 *  static deny-list (categorical metadata, not a free-text tag). */
export const CATEGORIES_WITHOUT_CUSTOM_INPUT: ReadonlySet<string> = new Set([
    TAG_CATEGORIES.rating
]);

/** Option in the per-chip section-reassignment popover. */
export interface CategoryOption {
    /** ``null`` = Auto (clear the override, derive from model). */
    value: string | null;
    /** Full label shown in the popover menu. */
    label: string;
    /** Compact label shown on the chip badge. */
    short: string;
}

/** Section-reassignment options offered on starred chips. ``Auto`` clears the
 *  override (derive from model); the others force the tag into that section's
 *  lead in the per-image prune grid. */
export const STARRED_CATEGORY_OPTIONS: readonly CategoryOption[] = [
    { value: null, label: 'Auto', short: 'Auto' },
    { value: TAG_CATEGORIES.character, label: 'Character', short: 'Char' },
    { value: TAG_CATEGORIES.general, label: 'General', short: 'Gen' }
];
