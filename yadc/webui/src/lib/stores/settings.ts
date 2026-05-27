import storable from '$lib/storable.js';
import { writable } from 'svelte/store';

export interface Settings {
    $version: number;
    thumbnailsPerRow: number;
    /** Browser notification preference: "unset" (never asked), "enabled", or "disabled". */
    notifications: 'unset' | 'enabled' | 'disabled';
}

export const settings = storable<Settings>('yadc/settings', {
    $version: 1,
    thumbnailsPerRow: 5,
    notifications: 'unset'
});

export const settingsDialog = writable<{
    open: boolean;
    tab: 'general' | 'environments' | 'security';
}>({
    open: false,
    tab: 'general'
});
