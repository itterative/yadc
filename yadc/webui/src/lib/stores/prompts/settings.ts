import storable from '$lib/storable.js';
import type { PromptGenFocus, PromptImageQuality } from './types';

/**
 * Persisted form settings for the prompt generator.
 *
 * Backed by ``storable`` (a Svelte writable store that mirrors itself
 * to localStorage) under the key ``yadc.prompts/formSettings``. The
 * ``$version`` field follows the convention used by the other settings
 * stores (``yadc/settings``, ``yadc/captionSettings``) so future shape
 * changes can ship a migration.
 *
 * The few-shot ``examples`` list is intentionally NOT persisted — its
 * ``image_data_url`` payloads are too large for localStorage's
 * ~5–10MB quota, and we don't currently track per-example provenance
 * (e.g. ``{type: "dataset", dataset, imageId}``) needed to re-fetch
 * them on reload. See the prompt-generator plan history for the
 * deferred IndexedDB follow-up.
 *
 * Refine-mode-only fields (``templateContent``, the template picker
 * selection) are also NOT persisted — they're ephemeral local state in
 * the host. On reload, the user re-selects a template (or pastes) to
 * populate the editor. The ``mode`` switch (New/Refine, an inline
 * toggle in the form) IS persisted so the user comes back to the same
 * selection. The Settings-tab fields — env / apiUrl / apiToken /
 * apiModelName plus the generation limits (``maxTokens``,
 * ``imageQuality``) — are persisted here too.
 */

export type PromptGenMode = 'generate' | 'refine';

export interface PromptFormSettings {
    $version: number;
    mode: PromptGenMode;
    env: string;
    apiUrl: string;
    apiToken: string;
    apiModelName: string;
    intent: string;
    focus: PromptGenFocus;
    /** Output token cap (client default of 4096 truncates longer templates). */
    maxTokens: number;
    /** Few-shot example image fidelity — OpenAI ``image_url.detail`` /
     *  Gemini ``mediaResolution``. */
    imageQuality: PromptImageQuality;
}

const DEFAULTS: PromptFormSettings = {
    $version: 3,
    mode: 'generate',
    env: 'default',
    apiUrl: '',
    apiToken: '',
    apiModelName: '',
    intent: '',
    focus: 'both',
    maxTokens: 16384,
    imageQuality: 'auto'
};

/** Migrate a stored object to the current shape (v3).
 *
 *  Re-applies the current defaults as the base, then overlays the
 *  stored fields so the user's existing settings are preserved.
 *  Missing fields (e.g. ``maxTokens`` / ``imageQuality`` added in v3,
 *  or ``mode`` added in v2) fall back to defaults; extra fields are
 *  dropped; ``$version`` is forced to the current value. Shape-agnostic
 *  so the same overlay works for any prior version. */
function migrateToCurrent(
    stored: Partial<PromptFormSettings> & { $version: number },
    _version: number
): PromptFormSettings {
    return { ...DEFAULTS, ...stored, $version: DEFAULTS.$version };
}

export const promptSettings = storable<PromptFormSettings>(
    'yadc/prompts/formSettings',
    DEFAULTS,
    migrateToCurrent
);

/** Reset to defaults and clear the localStorage entry. */
export function resetPromptSettings(): void {
    promptSettings.set({ ...DEFAULTS });
}
