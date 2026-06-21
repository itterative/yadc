import storable from '$lib/storable.js';
import type { PromptGenFocus } from './types';

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
 * populate the editor. Only the ``mode`` switch itself is persisted so
 * the user comes back to the same tab.
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
}

const DEFAULTS: PromptFormSettings = {
    $version: 2,
    mode: 'generate',
    env: 'default',
    apiUrl: '',
    apiToken: '',
    apiModelName: '',
    intent: '',
    focus: 'both'
};

/** Migrate a v1 stored object (no ``mode`` field) to v2.
 *
 *  Re-applies the current defaults as the base, then overlays the
 *  stored fields so the user's existing env / apiUrl / apiToken /
 *  apiModelName / intent / focus are preserved. The ``mode`` defaults
 *  to ``'generate'`` (the v1 behaviour). Bumps ``$version`` to 2.
 *
 *  ``_version`` is unused but matches the ``migrate`` signature that
 *  ``storable`` calls. The function is shape-agnostic: if a future
 *  version (v0 or v3+) needs migration to v2, the same overlay
 *  strategy works — missing fields fall back to defaults, extra
 *  fields are dropped, and the version is forced to 2. */
function migrateToV2(
    stored: Partial<PromptFormSettings> & { $version: number },
    _version: number
): PromptFormSettings {
    return { ...DEFAULTS, ...stored, $version: 2 };
}

export const promptSettings = storable<PromptFormSettings>(
    'yadc/prompts/formSettings',
    DEFAULTS,
    migrateToV2
);

/** Reset to defaults and clear the localStorage entry. */
export function resetPromptSettings(): void {
    promptSettings.set({ ...DEFAULTS });
}
