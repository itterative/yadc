/**
 * Browser notification helpers for long-running captioning jobs.
 *
 * Uses the Notification API to alert the user when captioning finishes
 * (or errors out) so they don't need to keep the tab in focus.
 *
 * Permission is requested when the user enables notifications in settings.
 * The preference is stored in the Settings store, not here — this module
 * only handles the actual dispatching.
 */

import { browser } from '$app/environment';
import { get } from 'svelte/store';
import { settings } from '$lib/stores/settings';
import { addToast, dismissToast } from '$lib/stores/toasts';

// --- Permission & support ---

/** Check if the browser supports notifications. */
export function notificationsSupported(): boolean {
    return browser && 'Notification' in window;
}

/** Current permission level (or "unsupported"). */
export function notificationPermission(): NotificationPermission | 'unsupported' {
    if (!notificationsSupported()) {
        return 'unsupported';
    }
    return Notification.permission;
}

/**
 * Request notification permission from the browser.
 * Returns the resulting permission string.
 */
export async function requestNotificationPermission(): Promise<NotificationPermission> {
    if (!notificationsSupported()) {
        return 'denied';
    }
    return Notification.requestPermission();
}

// --- Sending ---

/**
 * Send a browser notification.
 *
 * Silently no-ops if any of: notifications aren't supported, the user
 * hasn't enabled them in settings, browser permission isn't granted, or
 * the tab is in the foreground (unless `always` is true).
 *
 * Returns the Notification object if one was created, or null.
 */
export function sendNotification(opts: {
    title: string;
    body?: string;
    tag?: string;
    /** If true, show even when the tab is focused. */
    always?: boolean;
}): Notification | null {
    if (!notificationsSupported()) {
        return null;
    }
    if (get(settings).notifications !== 'enabled') {
        return null;
    }
    if (Notification.permission !== 'granted') {
        return null;
    }

    // Don't bother if the tab is visible
    if (!opts.always && document.visibilityState === 'visible') {
        return null;
    }

    return new Notification(`yadc - ${opts.title}`, {
        body: opts.body ?? '',
        tag: opts.tag,
        icon: '/android-chrome-192x192.png'
    });
}

// --- First-use prompt ---

/** Tracks whether the one-time prompt has already been shown this session. */
let promptShown = false;

/**
 * Show a one-time toast prompting the user to enable browser notifications.
 *
 * No-ops if any of: notifications aren't supported, they're already enabled
 * in settings, or the prompt has already been shown this session.
 * The toast has an "Enable" action button that requests permission and updates
 * settings; the user can also dismiss it normally.
 */
export function promptNotificationsOnce(): void {
    if (promptShown) {
        return;
    }
    if (!notificationsSupported()) {
        return;
    }
    if (get(settings).notifications !== 'unset') {
        return;
    }

    promptShown = true;

    const toastId = addToast({
        message: 'Enable browser notifications to get alerted when captioning finishes?',
        variant: 'info',
        duration: 15_000,
        actions: [
            {
                label: 'Enable',
                handler: () => {
                    dismissToast(toastId);
                    requestNotificationPermission().then((perm) => {
                        if (perm === 'granted') {
                            settings.update((s) => ({ ...s, notifications: 'enabled' }));
                        } else {
                            settings.update((s) => ({ ...s, notifications: 'disabled' }));
                        }
                    });
                }
            },
            {
                label: "Don't Ask Again",
                handler: () => {
                    dismissToast(toastId);
                    settings.update((s) => ({ ...s, notifications: 'disabled' }));
                }
            }
        ]
    });
}
