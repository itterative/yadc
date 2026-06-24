import storable from '$lib/storable.js';

/** Persisted tagger UI settings. The thresholds are override candidates —
 *  ``null`` means "use the server config default" (the batch panel omits
 *  them from the request body). Save options mirror ``TagSaveOptions``. */
export interface TagSettings {
    $version: number;
    /** ``null`` = server default (omit from request). */
    ratingThreshold: number | null;
    generalThreshold: number | null;
    characterThreshold: number | null;
    /** Replace underscores with spaces in tag names (kaomojis preserved). */
    replaceUnderscores: boolean;
    saveMode: 'none' | 'draft' | 'extras';
    draftName: string;
    draftFormat: string;
    /** Batch-only: skip images that already have the target save artifact. */
    overwrite: boolean;
}

/** Canonical wd-tagger defaults — what the server falls back to when a
 *  threshold isn't overridden. Used as the diff-dot baseline in the batch
 *  panel (the server's global ``Configuration`` is the true source, but it
 *  isn't exposed per-dataset, and these are its defaults). */
export const CANONICAL_THRESHOLDS = {
    rating: 0.0,
    general: 0.35,
    character: 0.85
} as const;

export const tagSettings = storable<TagSettings>(
    'yadc/tagSettings',
    {
        $version: 3,
        ratingThreshold: null,
        generalThreshold: null,
        characterThreshold: null,
        replaceUnderscores: false,
        saveMode: 'draft',
        draftName: 'tags',
        draftFormat: 'comma',
        overwrite: false
    },
    // v2 → v3: add ``overwrite`` (defaults off — preserve existing tags by default).
    (data) => ({ ...(data as object), $version: 3, overwrite: false }) as TagSettings
);
