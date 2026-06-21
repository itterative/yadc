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
 */

export interface PromptFormSettings {
    $version: number;
    env: string;
    apiUrl: string;
    apiToken: string;
    apiModelName: string;
    intent: string;
    focus: PromptGenFocus;
}

export const promptSettings = storable<PromptFormSettings>('yadc/prompts/formSettings', {
    $version: 1,
    env: 'default',
    apiUrl: '',
    apiToken: '',
    apiModelName: '',
    intent: '',
    focus: 'both'
});

/** Reset to defaults and clear the localStorage entry. */
export function resetPromptSettings(): void {
    promptSettings.set({
        $version: 1,
        env: 'default',
        apiUrl: '',
        apiToken: '',
        apiModelName: '',
        intent: '',
        focus: 'both'
    });
}
