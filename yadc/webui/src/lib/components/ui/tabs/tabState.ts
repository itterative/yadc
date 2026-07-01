import { get } from 'svelte/store';
import storable from '$lib/storable.js';

interface TabStateMap {
    $version: number;
    /** ``storageId`` -> last active tab id, keyed by the consuming tab host. */
    tabs: Record<string, string>;
}

const tabState = storable<TabStateMap>('yadc/tabState', { $version: 1, tabs: {} }, null);

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
