import storable from '$lib/storable.js';
import { writable } from 'svelte/store';
import { z } from 'zod';

export interface Settings {
    $version: number;
    thumbnailsPerRow: number;
    /** Browser notification preference: "unset" (never asked), "enabled", or "disabled". */
    notifications: 'unset' | 'enabled' | 'disabled';
}

const SettingsSchema = z.object({
    $version: z.number(),
    thumbnailsPerRow: z.number(),
    notifications: z.enum(['unset', 'enabled', 'disabled'])
});

export const settings = storable(
    'yadc/settings',
    {
        $version: 1,
        thumbnailsPerRow: 5,
        notifications: 'unset'
    },
    null,
    SettingsSchema
);

export const settingsDialog = writable<{
    open: boolean;
    tab: 'general' | 'environments' | 'security';
}>({
    open: false,
    tab: 'general'
});
