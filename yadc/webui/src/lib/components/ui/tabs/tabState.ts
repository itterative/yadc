import { get } from 'svelte/store';
import storable from '$lib/storable.js';
import { z } from 'zod';

const TabStateSchema = z.object({
    $version: z.number(),
    /** ``storageId`` -> last active tab id, keyed by the consuming tab host. */
    tabs: z.record(z.string(), z.string()).default({})
});

const tabState = storable('yadc/tabState', { $version: 1, tabs: {} }, null, TabStateSchema);

/** Read the last-selected tab id for a given storage scope. Returns ``null``
 *  when unset. Callers are responsible for validating the id still exists in
 *  their tab list. */
export function loadStoredTabId(storageId: string): string | null {
    return get(tabState).tabs[storageId] ?? null;
}

/** Persist the active tab id under the given storage scope. */
export function saveStoredTabId(storageId: string, id: string): void {
    tabState.update((s) => ({ ...s, tabs: { ...s.tabs, [storageId]: id } }));
}
