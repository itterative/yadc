import storable from '$lib/storable.js';
import { z } from 'zod';

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
    /** Use per-tag optimal thresholds from the CSV instead of global category thresholds. */
    perTagThresholds: boolean;
    /** Which per-tag threshold column to use (best_threshold or best_recall). */
    perTagColumn: string;
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

const TagSettingsSchema = z.object({
    $version: z.number(),
    ratingThreshold: z.number().nullable(),
    generalThreshold: z.number().nullable(),
    characterThreshold: z.number().nullable(),
    replaceUnderscores: z.boolean(),
    saveMode: z.enum(['none', 'draft', 'extras']),
    draftName: z.string(),
    draftFormat: z.string(),
    overwrite: z.boolean(),
    perTagThresholds: z.boolean(),
    perTagColumn: z.string()
});

export const tagSettings = storable(
    'yadc/tagSettings',
    {
        $version: 4,
        ratingThreshold: null,
        generalThreshold: null,
        characterThreshold: null,
        replaceUnderscores: false,
        saveMode: 'draft',
        draftName: 'tags',
        draftFormat: 'comma',
        overwrite: false,
        perTagThresholds: false,
        perTagColumn: 'best_threshold'
    },
    // v3 → v4: add per-tag threshold toggle and column selector.
    (data) => ({ ...(data as object), $version: 4, perTagThresholds: false, perTagColumn: 'best_threshold' }) as TagSettings,
    TagSettingsSchema
);
